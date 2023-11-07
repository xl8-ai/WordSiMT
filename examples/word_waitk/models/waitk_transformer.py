"""
code from https://github.com/elbayadm/attn2d/blob/master/examples/waitk/models/waitk_transformer.py
"""
import random

import torch
from transformers import AutoModel, AutoTokenizer, XLNetTokenizerFast

from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    TransformerDecoder,
    TransformerModel,
)
from word_waitk.models import UnidirTransformerEncoder
from word_waitk.modules import WaitkTransformerDecoderLayer
from fairseq.modules.checkpoint_activations import checkpoint_wrapper


@register_model("waitk_transformer")
class WaitkTransformerModel(TransformerModel):
    """
    Waitk-Transformer with a uni-directional encoder
    """

    def __init__(self, args, encoder, decoder, lm, lm_tokenizer):
        super().__init__(args, encoder, decoder)
        self.lm = lm
        self.lm_tokenizer = lm_tokenizer
        self.tokens_with_space_symbol = (
            torch.tensor([i for t, i in self.lm_tokenizer.vocab.items() if "▁" in t])
            if isinstance(self.lm_tokenizer, XLNetTokenizerFast)
            else None
        )

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, lm_input=None, **kwargs
    ):
        if self.lm is not None:
            lm_pad_index = None
            if getattr(self.lm_tokenizer, "pad_token_id", None) is not None:
                lm_pad_index = self.lm_tokenizer.pad_token_id
            elif getattr(self.lm_tokenizer, "bos_token_id", None) is not None:
                # GPT2 does not have pad token.
                lm_pad_index = self.lm_tokenizer.bos_token_id
            lm_padding_mask = lm_input.eq(lm_pad_index)
            extra_args = {}
            if isinstance(self.lm_tokenizer, XLNetTokenizerFast):
                if self.tokens_with_space_symbol.device != lm_input.device:
                    self.tokens_with_space_symbol = self.tokens_with_space_symbol.to(
                        device=lm_input.device
                    )
                word_starts = torch.isin(lm_input, self.tokens_with_space_symbol).int()
                word_idxs = word_starts.cumsum(dim=1).unsqueeze(2)
                word_idxs[word_idxs > 0] -= 1
                extra_args["perm_mask"] = (
                    word_idxs < word_idxs.transpose(1, 2)
                ).float()
            lm_out = self.lm(
                lm_input,
                output_hidden_states=True,
                attention_mask=(~lm_padding_mask).int(),
                **extra_args
            )
            lm_out = lm_out.hidden_states[-1]
            lm_out = lm_out.permute(1, 0, 2).contiguous()
            lm_out = {
                "lm_input": lm_input,
                "lm_out": lm_out,
                "lm_padding_mask": lm_padding_mask,
            }
        else:
            lm_out = {"lm_input": None, "lm_out": None, "lm_padding_mask": None}
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, lm_out=lm_out, **kwargs
        )
        decoder_out = self.decoder.forward_train(
            prev_output_tokens, encoder_out=encoder_out, lm_out=lm_out, **kwargs
        )
        return decoder_out

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument("--waitk", type=int, help="wait-k for incremental reading")
        parser.add_argument(
            "--min-waitk", type=int, help="wait-k for incremental reading"
        )
        parser.add_argument(
            "--max-waitk", type=int, help="wait-k for incremental reading"
        )
        parser.add_argument(
            "--multi-waitk",
            action="store_true",
            help="Use multiple k values in waitk traiing",
            default=False,
        )
        parser.add_argument(
            "--word-waitk",
            action="store_true",
            help="Apply word-level waitk",
            default=False,
        )
        parser.add_argument(
            "--encoder-token-level-causal-mask",
            action="store_true",
            help="Apply token-level causal mask in encoder for word-level waitk",
            default=False,
        )
        parser.add_argument(
            "--skip-loading-from-huggingface",
            default=False,
            action="store_true",
            help="skip huggingface Bert Model loading",
        )
        parser.add_argument(
            "--leading-space-symbol",
            default=False,
            action="store_true",
            help="Set true if space is leading and the model is word-waitk.",
        )
        parser.add_argument(
            "--encoder-bpe-symbol",
            default="\u2581",
            help="BPE symbol applied to encoder.",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        if not hasattr(args, "language_model_name"):
            args.language_model_name = None
        if args.language_model_name is not None:
            if len(task.datasets) > 0:
                lm_tokenizer = next(iter(task.datasets.values())).lm_tokenizer
            else:
                lm_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name)

            lm = AutoModel.from_pretrained(args.language_model_name)

            args.lm_out_dim = (
                getattr(lm.config, "hidden_size", None)
                or getattr(lm.config, "d_model", None)
                or getattr(lm.config, "n_embd")
            )
        else:
            lm = lm_tokenizer = None

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing
        if not getattr(args, "consnmt", False):
            encoder = cls.build_encoder(
                args, src_dict, encoder_embed_tokens, lm_tokenizer=lm_tokenizer
            )
        else:
            encoder = cls.build_encoder(
                args, src_dict, encoder_embed_tokens, decoder_embed_tokens
            )
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)

        return cls(args, encoder, decoder, lm, lm_tokenizer)

    @classmethod
    def build_encoder(
        cls, args, src_dict, embed_tokens, decoder_embed_tokens=None, lm_tokenizer=None
    ):
        """
        decoder_embed_tokens is for LeCA model.
        It is a shared parameter with the embed_tokens of decoder.
        """
        return UnidirTransformerEncoder(
            args, src_dict, embed_tokens, lm_tokenizer=lm_tokenizer
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return WaitkTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )


class WaitkTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.waitk = args.waitk
        self.min_waitk = args.min_waitk
        self.max_waitk = args.max_waitk
        self.multi_waitk = args.multi_waitk
        self.word_waitk = getattr(args, "word_waitk", False)
        self.tokens_with_space_symbol = torch.tensor(
            [
                self.dictionary.indices[tok]
                for tok in self.dictionary.symbols
                if "\u2581" in tok
            ]
        )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = WaitkTransformerDecoderLayer(args, no_encoder_attn)
        if getattr(args, "checkpoint_activations", False):
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        return layer

    def predict(self, x):
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        # project back to size of vocabulary
        x = self.output_projection(x)
        return x

    def get_attention_mask(
        self,
        src_len,
        waitk=None,
        prev_output_tokens=None,
        src_cumsum=None,
        lm_cumsum=None,
    ):
        batch_size, tgt_len = prev_output_tokens.size()
        if self.word_waitk:
            assert src_cumsum is not None
            max_src_words = torch.max(src_cumsum) + 1

            if waitk is None:
                if self.multi_waitk:
                    assert self.min_waitk <= self.max_waitk
                    waitk = random.randint(
                        min(self.min_waitk, max_src_words),
                        min(max_src_words, self.max_waitk),
                    )
                else:
                    waitk = self.waitk
            if self.tokens_with_space_symbol.device != prev_output_tokens.device:
                self.tokens_with_space_symbol = self.tokens_with_space_symbol.to(
                    prev_output_tokens
                )
            sp_symbol_locations = torch.isin(
                prev_output_tokens, self.tokens_with_space_symbol
            )
            tgt_word_end_mask = torch.roll(sp_symbol_locations, shifts=-1, dims=1)
            cumsum = (
                tgt_word_end_mask.cumsum(dim=1) - tgt_word_end_mask.int()
            ).unsqueeze(2)
            encoder_attn_mask = (
                torch.zeros((batch_size, tgt_len, src_len), device=src_cumsum.device)
                .float()
                .masked_fill_(
                    cumsum + waitk - 1 < src_cumsum.transpose(1, 2), float("-inf")
                )
            )
        else:
            if waitk is None:
                if self.multi_waitk:
                    assert self.min_waitk <= self.max_waitk
                    waitk = random.randint(
                        min(self.min_waitk, src_len), min(src_len, self.max_waitk)
                    )
                else:
                    waitk = self.waitk

            if waitk < src_len:
                if src_cumsum is None:
                    encoder_attn_mask = torch.triu(
                        utils.fill_with_neg_inf(
                            prev_output_tokens.new(tgt_len, src_len)
                        ),
                        waitk,
                    )
                    if waitk <= 0:
                        encoder_attn_mask[:, 0] = 0
                else:  # token-level wait-k
                    cumsum = (
                        torch.arange(tgt_len, device=src_cumsum.device)
                        .repeat(batch_size, 1, 1)
                        .transpose(1, 2)
                    )
                    encoder_attn_mask = (
                        torch.zeros(
                            (batch_size, tgt_len, src_len), device=src_cumsum.device
                        )
                        .float()
                        .masked_fill_(
                            cumsum + waitk - 1 < src_cumsum.transpose(1, 2),
                            float("-inf"),
                        )
                    )
            else:
                encoder_attn_mask = None

        if lm_cumsum is not None and encoder_attn_mask is not None:
            lm_len = lm_cumsum.size(1)
            if encoder_attn_mask.dim() == 3:
                lm_attn_mask = (
                    torch.zeros((batch_size, tgt_len, lm_len), device=lm_cumsum.device)
                    .float()
                    .masked_fill_(
                        cumsum + waitk - 1 < lm_cumsum.transpose(1, 2), float("-inf")
                    )
                )
            else:
                cumsum = (
                    torch.arange(tgt_len, device=src_cumsum.device)
                    .repeat(batch_size, 1, 1)
                    .transpose(1, 2)
                )
                lm_attn_mask = (
                    torch.zeros((batch_size, tgt_len, lm_len), device=lm_cumsum.device)
                    .float()
                    .masked_fill_(
                        cumsum + waitk - 1 < lm_cumsum.transpose(1, 2), float("-inf")
                    )
                )
        else:
            lm_attn_mask = None

        return encoder_attn_mask, lm_attn_mask

    def forward_train(
        self, prev_output_tokens, encoder_out=None, lm_out=None, **kwargs
    ):
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
        if positions is not None:
            x += positions
        x = self.dropout_module(x)

        self_attn_padding_mask = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        # encoder attn mask following the reading/writing schedule len_tgt x len_src
        encoder_states = encoder_out["encoder_out"][0]  # len_src, B, C
        encoder_attn_mask, lm_attn_mask = self.get_attention_mask(
            encoder_states.size(0),
            prev_output_tokens=prev_output_tokens,
            src_cumsum=encoder_out.get("word_end_cumsum", None),
            lm_cumsum=encoder_out.get("lm_cumsum", None),
        )

        if encoder_attn_mask is not None:
            encoder_attn_mask = encoder_attn_mask.to(x)
        if lm_attn_mask is not None:
            lm_attn_mask = lm_attn_mask.to(x)

        # decoder layers
        for e, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                encoder_states,
                encoder_out["encoder_padding_mask"][0],
                encoder_attn_mask=encoder_attn_mask,
                self_attn_mask=self.buffered_future_mask(x),
                self_attn_padding_mask=self_attn_padding_mask,
                lm_out=lm_out["lm_out"],
                lm_padding_mask=lm_out["lm_padding_mask"],
                lm_attn_mask=lm_attn_mask,
            )

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        x = self.predict(x)
        return x, {"attn": attn}

    def forward(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        waitk=1024,
        step=None,
        static_encoder_attn_kv=False,
        lm_out=None,
        **kwargs
    ):
        # Evaluation.
        positions = (
            self.embed_positions(
                prev_output_tokens,
                incremental_state=incremental_state,
                timestep=step,
            )
            if self.embed_positions is not None
            else None
        )
        if encoder_out is not None and (
            not self.word_waitk or encoder_out.get("word_end_cumsum", None) is not None
        ):
            encoder_attn_mask, lm_attn_mask = self.get_attention_mask(
                encoder_out["encoder_out"][0].size(0),
                waitk=waitk,
                prev_output_tokens=prev_output_tokens,
                src_cumsum=encoder_out.get("word_end_cumsum", None),
                lm_cumsum=encoder_out.get("lm_cumsum", None),
            )
        else:
            encoder_attn_mask = lm_attn_mask = None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
            if encoder_attn_mask is not None:
                if encoder_attn_mask.dim() == 2:
                    encoder_attn_mask = encoder_attn_mask[-1:, :]
                else:
                    encoder_attn_mask = encoder_attn_mask[:, -1:, :]
            if lm_attn_mask is not None:
                if lm_attn_mask.dim() == 2:
                    lm_attn_mask = lm_attn_mask[-1:, :]
                else:
                    lm_attn_mask = lm_attn_mask[:, -1:, :]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = self.dropout_module(x)

        self_attn_padding_mask = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        if encoder_attn_mask is not None:
            encoder_attn_mask = encoder_attn_mask.to(x)

        if lm_attn_mask is not None:
            lm_attn_mask = lm_attn_mask.to(x)

        # decoder layers
        for e, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                encoder_attn_mask=encoder_attn_mask,
                incremental_state=incremental_state,
                self_attn_mask=self.buffered_future_mask(x)
                if incremental_state is None
                else None,
                static_encoder_attn_kv=static_encoder_attn_kv,
                self_attn_padding_mask=self_attn_padding_mask,
                lm_out=lm_out["lm_out"] if lm_out is not None else None,
                lm_padding_mask=lm_out["lm_padding_mask"]
                if lm_out is not None
                else None,
                lm_attn_mask=lm_attn_mask,
            )

        if self.layer_norm:
            x = self.layer_norm(x)

        # if incremental_state is not None:
        # Project only the last token
        x = x[-1:]

        if static_encoder_attn_kv:
            if encoder_attn_mask is None:  # token-level, wait-k bigger than src_len
                token_context = [
                    encoder_out["encoder_out"][0].size(0)
                ] * prev_output_tokens.size(0)
            else:
                token_context_tensor = encoder_attn_mask.size(-1) - torch.argmax(
                    encoder_attn_mask.flip(-1), dim=-1
                )
                if encoder_attn_mask.dim() == 3:
                    token_context = token_context_tensor.squeeze(-1).tolist()
                else:
                    token_context = (
                        token_context_tensor.tolist() * prev_output_tokens.size(0)
                    )
        else:
            token_context = None

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        x = self.predict(x)
        return x, {"attn": attn, "token_context": token_context}


@register_model_architecture("waitk_transformer", "waitk_transformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    args.waitk = getattr(args, "waitk", 1024)  # wait-until-end
    args.min_waitk = getattr(args, "min_waitk", 1)
    args.max_waitk = getattr(args, "max_waitk", 1024)


@register_model_architecture("waitk_transformer", "waitk_transformer_small")
def waitk_transformer_small(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.dropout = getattr(args, "dropout", 0.3)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    base_architecture(args)


@register_model_architecture("waitk_transformer", "waitk_transformer_iwslt_de_en")
def waitk_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)


@register_model_architecture("waitk_transformer", "waitk_transformer_base")
def waitk_transformer_base(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.dropout = getattr(args, "dropout", 0.3)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    base_architecture(args)


@register_model_architecture(
    "waitk_transformer", "waitk_transformer_vaswani_wmt_en_de_big"
)
def waitk_transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


@register_model_architecture("waitk_transformer", "waitk_transformer_big")
def waitk_transformer_big(args):
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    waitk_transformer_vaswani_wmt_en_de_big(args)
