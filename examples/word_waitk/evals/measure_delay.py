import os.path as osp
import numpy as np
import re
from typing import List
import sacrebleu


def get_al(ctxs: List[int], src_len: int, gamma: float):
    """Calculate average lagging."""
    als = []
    for t, c in enumerate(ctxs):
        als.append(c - t / gamma)
        if c >= src_len:
            break
    return sum(als)/float(len(als))


def get_context(
    src: List[str],
    hyp: List[str],
    waitk: int,
    word_waitk: bool = True,
    return_word_index: bool = False):
    """Get context from source tokens and hypo tokens.
    
    if return_word_index is True, context is defined as number of source words read before writing a target word.
        for token-level wait-k model, context is the minimum number of words that contain
        min(tgt_token_idx + waitk, num_src_tokens) tokens.
        for word-level wait-k model, context is min(tgt_word_idx + waitk, num_src_words).

    if return_word_index is False, context is defined as number of source tokens read before writing a target token.
        for token-level wait-k model, context is min(tgt_token_idx + waitk, num_src_tokens).
        for word-level wait-k model, context is the number of tokens in min(tgt_word_idx + waitk, num_src_words).
    """
    context = []
    end_of_word_mask = np.array([1 if "▁" in s else 0 for s in src])
    src_token_to_num_words = (np.cumsum(end_of_word_mask) - end_of_word_mask + 1).tolist()
    src_num_words_to_max_tokens = {}
    for t, n in enumerate(src_token_to_num_words):
        src_num_words_to_max_tokens[n] = t + 1
    num_src_words = src_token_to_num_words[-1]

    if return_word_index:
        if word_waitk:
            count = 0
            for h in hyp:
                if "▁" in h:
                    context.append(min(count + waitk, num_src_words))
                    count += 1
        else:
            for i, h in enumerate(hyp):
                if "▁" in h:
                    context_src_token_idx = min(i + waitk, len(src)) - 1
                    context.append(src_token_to_num_words[context_src_token_idx])
    else:  # return token index
        if word_waitk:
            count = 0
            for h in hyp:
                context.append(src_num_words_to_max_tokens[min(count + waitk, num_src_words)])
                if "▁" in h:
                    count += 1
        else:
            for i, h in enumerate(hyp):
                context.append(min(i + waitk, len(src)))

    return context


def get_delays(res: str, waitk: int, lang: str, encoder_bpe_symbol: str, reference_file: str):
    """Parse tokenized sources, targets and hypos and calculate delays."""
    src_lines = {}
    tgt_lines = {}
    hyp_lines = {}
    token_contexts = {}
    word_contexts = {}
    waitk = int(waitk)
    if osp.exists(res):
        with open(res, 'r') as f:
            for line in f:
                if line.startswith('S-'):
                    line = line.split('S-', 1)[-1]
                    sid = int(line.split()[0])
                    src_lines[sid] = line.split("\t")[-1].strip()
                elif line.startswith('T-'):
                    line = line.split('T-', 1)[-1]
                    sid = int(line.split()[0])
                    tgt_lines[sid] = line.split("\t")[-1].strip()
                elif line.startswith('H-'):
                    line = line.split('H-', 1)[-1]
                    sid = int(line.split()[0])
                    hyp_lines[sid] = line.split("\t")[-1].strip()
                elif line.startswith('TC-'):
                    line = line.split('TC-', 1)[-1]
                    sid = int(line.split()[0])
                    token_contexts[sid] = [
                        int(tc) for tc in line.split()[1:]
                    ]
                elif line.startswith('WC-'):
                    line = line.split('WC-', 1)[-1]
                    sid = int(line.split()[0])
                    word_contexts[sid] = [
                        int(wc) for wc in line.split()[1:]
                    ]
                
            # Parse the first line
            f.seek(0)
            try:
                args_dict = eval("|".join(f.readline().split("|")[3:]))
            except:
                f.seek(0)
                args_dict = eval(f.readline())
            word_waitk = args_dict["task"]["word_waitk"]

        if "w_con" in res:
            with open(res.replace("w_con", "nocon"), 'r') as f:
                src_lines = {}
                for line in f:
                    if line.startswith('S-'):
                        line = line.split('S-', 1)[-1]
                        sid = line.split()[0]
                        src_lines[sid] = line.split("\t")[-1].strip()

        token_laggings = []
        word_laggings = []

        # print(f"Evaluate {'word' if word_waitk else 'token'} model...")

        for sid in src_lines:
            if encoder_bpe_symbol == "Ġ":
                src_tokens = src_lines[sid].split() + ["Ġ"]  # <eos>
            else:
                src_tokens = src_lines[sid].split() + ["▁"]  # <eos>
            hypo_tokens = hyp_lines[sid].split() + ["▁"]  # <eos>
            if sid in token_contexts:
                token_context = token_contexts[sid]
            else:
                token_context = get_context(src_tokens, hypo_tokens, waitk, word_waitk, False)
            if sid in word_contexts:
                word_context = word_contexts[sid]
            else:
                word_context = get_context(src_tokens, hypo_tokens, waitk, word_waitk, True)
            if encoder_bpe_symbol == "Ġ":  # The first token of GPT2 does not have Ġ.
                num_src_words = len([token for token in src_tokens if "Ġ" in token]) + 1
            else:
                num_src_words = len([token for token in src_tokens if "▁" in token])
            num_hypo_words = len([token for token in hypo_tokens if "▁" in token])
            token_lagging = get_al(token_context, len(src_tokens), len(hypo_tokens)/len(src_tokens))
            if not num_hypo_words:
                word_lagging = num_src_words
            else:
                word_lagging = get_al(word_context, num_src_words, num_hypo_words/num_src_words)
            
            token_laggings.append(token_lagging)
            word_laggings.append(word_lagging)

            hypo = hyp_lines[sid].replace(" ","").replace("▁", " ").strip()

        tal = np.mean(np.array(token_laggings))
        wal = np.mean(np.array(word_laggings))

        # Since python3, dictionaries preserve order. But just in case...
        with open(reference_file) as f:
            reference_lines = [l.strip() for l in f.readlines()]
        hyp_lines_list = [v for k,v in sorted(hyp_lines.items(), key=lambda x: x[0])]

        assert len(reference_lines) == len(hyp_lines_list)
        bleu = sacrebleu.corpus_bleu(
            [hyp_line.replace(" ","").replace("▁", " ").strip() for hyp_line in hyp_lines_list],
            [reference_lines],
            tokenize="intl",
            use_effective_order=True
            ).score

        return (tal, wal, bleu)
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--generated-file', '-g', type=str, required=True,
    help="Path to the generate-valid.txt file. Hypos should be tokens, not detokenized strings."
        "The file can be generated by following command:"
        "fairseq-generate {data_path} --path {model_file_path} --max-tokens 3000 --gen-subset valid \ "
        "--scoring sacrebleu--results-path {result_path} --source-lang {source_lang} --target-lang {target_lang} \ "
        "--sacrebleu-tokenizer intl --task waitk_translation --eval-waitk {waitk} --fp16 --beam 1")
    parser.add_argument('--waitk', '-k', type=float, required=True, help="Evaluation wait-k value.")
    parser.add_argument('--lang', '-l', type=str, required=True, help="Target language.")
    parser.add_argument('--encoder-bpe-symbol', type=str, default="▁", help="Encoder bpe token.")
    parser.add_argument("--reference-file", type=str, required=True, help="Path to the target file")
    args = parser.parse_args()
    results = get_delays(args.generated_file, args.waitk, args.lang, args.encoder_bpe_symbol, args.reference_file)
    # print("Evaluating with k =", args.waitk)
    print(*results)
