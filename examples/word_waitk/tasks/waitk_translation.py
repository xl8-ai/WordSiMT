"""
code from https://github.com/elbayadm/attn2d/blob/master/examples/waitk/tasks/waitk_task.py
"""
import itertools
import logging
import os
import pickle as pkl
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch.cuda import current_device
from transformers import AutoTokenizer

from fairseq import search, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    indexed_dataset,
)
from fairseq.tasks import register_task
from fairseq.tasks.translation import (
    TranslationConfig,
    TranslationTask,
)
from word_waitk.data import LanguagePairWaitkDataset
logger = logging.getLogger(__name__)


def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    lm_model_name=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []
    src_lm_datasets = None if lm_model_name is None else []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
            if src_lm_datasets is not None:
                lm_prefix = os.path.join(
                    data_path, "{}.lm.{}-{}.".format(split_k, src, tgt)
                )
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
            if src_lm_datasets is not None:
                lm_prefix = os.path.join(
                    data_path, "{}.lm.{}-{}.".format(split_k, tgt, src)
                )
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        if src_lm_datasets is not None:
            src_lm_dataset = data_utils.load_indexed_dataset(
                lm_prefix + src, None, dataset_impl
            )
            src_lm_datasets.append(src_lm_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
        if src_lm_datasets is not None:
            src_lm_datasets = src_lm_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None
        if src_lm_datasets is not None:
            src_lm_datasets = ConcatDataset(src_lm_datasets, sample_ratios)

    lm_tokenizer = None
    if lm_model_name is not None:
        lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_name)

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairWaitkDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        src_lm=src_lm_datasets,
        src_lm_sizes=None if src_lm_datasets is None else src_lm_datasets.sizes,
        lm_tokenizer=lm_tokenizer,
    )


@dataclass
class WaitkTranslationConfig(TranslationConfig):
    eval_waitk: int = field(default=9, metadata={"help": "waitk value for evaluation"})
    word_waitk: bool = field(default=False, metadata={"help": "Apply word-level waitk"})

    # options for warmup
    warmup_from_nmt: bool = field(
        default=False, metadata={"help": "warmup from pre-trained NMT model"}
    )
    warmup_nmt_file: str = field(
        default="checkpoint_nmt.pt",
        metadata={"help": "pre-trained NMT model fine name for warmup training"},
    )

    # LM_NMT args
    language_model_name: Optional[str] = field(default=None)
    encoder_ratio: float = field(default=1.0)
    lm_ratio: float = field(default=1.0)
    finetune_lm: bool = field(default=False)
    mask_cls_sep: bool = field(default=False)
    lm_gates: List[int] = field(default=(1, 1, 1, 1, 1, 1))
    encoder_lm_dropout: bool = field(default=False)
    encoder_lm_dropout_ratio: float = field(default=0.25)
    lm_output_layer: int = field(default=-1)
    encoder_lm_mixup: bool = field(default=False)
    decoder_no_lm: bool = field(default=False)


@register_task("waitk_translation", dataclass=WaitkTranslationConfig)
class WaitkTranslation(TranslationTask):
    cfg: WaitkTranslationConfig

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=False,#self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            lm_model_name=getattr(self.cfg, "language_model_name", None),
        )

    def build_dataset_for_inference(
        self,
        src_tokens,
        src_lengths,
        constraints=None,
        prefix_tokens=None,
        append_eos_to_target=False,
        src_lm_datasets=None,
        src_lm_sizes=None,
        lm_tokenizer=None,
    ):
        return LanguagePairWaitkDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            left_pad_source=False,#self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
            tgt=prefix_tokens,
            append_eos_to_target=append_eos_to_target,
            training=False,
            src_lm=src_lm_datasets,
            src_lm_sizes=src_lm_sizes,
            lm_tokenizer=lm_tokenizer,
        )

    def build_generator(
        self,
        models,
        args,
        word_waitk=False,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        prefix_allowed_tokens_fn=None,
    ):
        from word_waitk.generators import SequenceGenerator
        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if word_waitk:
            extra_gen_cls_kwargs["word_waitk"] = True

        return SequenceGenerator(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 1),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            src_dict=self.source_dictionary,
            waitk=self.cfg.eval_waitk,
            **extra_gen_cls_kwargs,
        )

    def inference_step(
        self,
        generator,
        models,
        sample,
        prefix_tokens=None,
        constraints=None,
        is_eos=False,
    ):
        kwargs = {}
        with torch.no_grad():
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
                **kwargs,
            )
