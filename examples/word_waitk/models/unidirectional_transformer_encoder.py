from typing import Dict, List, Optional

import torch
from torch import Tensor

from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.models.transformer import Embedding, TransformerEncoder
from word_waitk.modules import UnidirTransformerEncoderLayer
from fairseq.modules.checkpoint_activations import checkpoint_wrapper


@with_incremental_state
class UnidirTransformerEncoder(TransformerEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens, lm_tokenizer=None):
        super().__init__(args, dictionary, embed_tokens)
        self._future_mask = torch.empty(0)
        self.word_waitk = getattr(args, "word_waitk", False)
        self.token_level_causal_mask = getattr(
            args, "encoder_token_level_causal_mask", False
        )
        self.encoder_bpe_symbol = getattr(args, "encoder_bpe_symbol", "\u2581")
        self.tokens_with_space_symbol = torch.tensor(
            [self.dictionary.eos_index]
            + [
                self.dictionary.indices[tok]
                for tok in self.dictionary.symbols
                if self.encoder_bpe_symbol in tok
            ]
        )
        self.leading_space_symbol = getattr(args, "leading_space_symbol", False)

        if lm_tokenizer is not None:
            lm_word_starts = []
            self.space_symbol = (
                "\u2581"
                if args.language_model_name in ["facebook/xglm-564M"]
                else "Ġ"
            )
            for k, v in lm_tokenizer.vocab.items():
                if self.space_symbol in k:
                    lm_word_starts.append(v)
            self.lm_word_starts = torch.tensor(lm_word_starts)
            self.language_model_name = args.language_model_name
        else:
            self.lm_word_starts = None
            self.language_model_name = None

        self.lm_tokenizer = lm_tokenizer

    def build_encoder_layer(self, args):
        layer = UnidirTransformerEncoderLayer(args)
        if getattr(args, "checkpoint_activations", False):
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        return layer

    def forward(
        self,
        src_tokens,
        src_lengths=None,
        return_all_hiddens=False,
        incremental_state=None,
        mask=None,
        step=None,
        lm_out=None,
        **kwargs
    ):
        """
        Args: src_tokens (batch, src_len)
              src_lengths (batch)
        Returns:
            dict: - **encoder_out** (src_len, batch, embed_dim)
                  - **encoder_padding_mask**  (batch, src_len)
        """
        positions = (
            self.embed_positions(
                src_tokens,
                incremental_state=incremental_state,
                timestep=step,
            )
            if self.embed_positions is not None
            else None
        )

        x = encoder_embedding = self.embed_scale * self.embed_tokens(src_tokens)
        if positions is not None:
            x = encoder_embedding + positions

        x = self.dropout_module(x)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        # account for padding while computing the representation
        if encoder_padding_mask is not None:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        self_attn_mask, cumsum = None, None
        if incremental_state is None:
            self_attn_mask, cumsum = self.buffered_future_mask(
                x, src_tokens
            )
        if (
            incremental_state is None
            and lm_out is not None
            and lm_out["lm_input"] is not None
        ):
            if cumsum is None:  # token-waitk
                bsz, src_len = src_tokens.size()
                cumsum = (
                    torch.arange(src_len, device=lm_out["lm_input"].device)
                    .repeat(bsz, 1, 1)
                    .transpose(1, 2)
                )
            lm_attn_mask, lm_cumsum = self.get_lm_attn_mask_and_cumsum(
                x, lm_out["lm_input"], cumsum
            )
        else:
            lm_attn_mask = lm_cumsum = None

        # encoder layers
        for layer in self.layers:
            # Make the encoder unidirectional
            x = layer(
                x,
                encoder_padding_mask if has_pads else None,
                self_attn_mask=self_attn_mask,
                incremental_state=incremental_state,
                lm_out=lm_out["lm_out"] if lm_out is not None else None,
                lm_padding_mask=lm_out["lm_padding_mask"]
                if lm_out is not None
                else None,
                lm_attn_mask=lm_attn_mask,
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [src_tokens],
            "src_lengths": [],
            "word_end_cumsum": cumsum,
            "lm_cumsum": lm_cumsum,
        }

    def buffered_future_mask(self, tensor, src_tokens):
        dim = tensor.size(0)
        batch_size = tensor.size(1)
        if self.word_waitk:
            # self attention mask for word-level waitk.
            if self.tokens_with_space_symbol.device != src_tokens.device:
                self.tokens_with_space_symbol = self.tokens_with_space_symbol.to(
                    src_tokens
                )
            sp_symbol_locations = torch.isin(
                src_tokens, self.tokens_with_space_symbol
            ).int()
            if self.leading_space_symbol:
                # sp_symbol_locations are starts of words.
                if self.encoder_bpe_symbol == "Ġ":  # Encoded with GPT2
                    sp_symbol_locations[:, 0] = 1
                cumsum = sp_symbol_locations.cumsum(dim=1).unsqueeze(2)
                cumsum[cumsum > 0] -= 1
            else:
                # sp_symbol_locations are ends of words.
                cumsum = (
                    sp_symbol_locations.cumsum(dim=1) - sp_symbol_locations
                ).unsqueeze(2)
            if self.token_level_causal_mask:
                mask = torch.triu(
                    utils.fill_with_neg_inf(
                        torch.zeros(
                            [dim, dim], device=tensor.device, dtype=tensor.dtype
                        )
                    ),
                    1,
                )
            else:
                mask = torch.zeros(
                    (batch_size, dim, dim), device=tensor.device, dtype=tensor.dtype
                ).masked_fill_(cumsum < cumsum.transpose(1, 2), float("-inf"))

            return mask, cumsum

        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)

        return self._future_mask[:dim, :dim], None

    def get_lm_attn_mask_and_cumsum(self, tensor, lm_tokens, src_cumsum):
        batch_size, lm_len = lm_tokens.size()
        src_len = src_cumsum.size(1)
        if self.word_waitk:
            if self.lm_word_starts.device != lm_tokens.device:
                self.lm_word_starts = self.lm_word_starts.to(lm_tokens)
            word_start_mask = torch.isin(lm_tokens, self.lm_word_starts).int()
            if self.space_symbol == "Ġ":
                # In GPT2 tokenizer, the first word is not prepended with "Ġ".
                # This condition may also apply to OPT tokenizer which adds the prefix </s>, e.g., ['</s>', 'Several', 'Ġyears']
                word_start_mask[:, 0] = 1
            lm_cumsum = word_start_mask.cumsum(dim=1)
            lm_cumsum[
                lm_cumsum > 0
            ] -= 1  # The first word may contain eos when the source was encoded with XGLM.
        else:
            token_start_mask = torch.ones_like(lm_tokens)
            if "xglm" in self.language_model_name:
                token_start_mask[:, :2] = 0  # The first token is eos in XGLM.
            else:
                token_start_mask[:, 0] = 0
            lm_cumsum = token_start_mask.cumsum(dim=1)

        lm_cumsum = lm_cumsum.unsqueeze(2)
        lm_attn_mask = torch.zeros(
            (batch_size, src_len, lm_len), device=tensor.device, dtype=tensor.dtype
        ).masked_fill_(src_cumsum < lm_cumsum.transpose(1, 2), float("-inf"))

        return lm_attn_mask, lm_cumsum

    def reorder_encoder_out(
        self, encoder_out: Dict[str, List[Tensor]], new_order, lm_out=None
    ):
        encoder_states = super().reorder_encoder_out(encoder_out, new_order)

        if encoder_out.get("word_end_cumsum") is not None:
            encoder_states["word_end_cumsum"] = encoder_out[
                "word_end_cumsum"
            ].index_select(0, new_order)
        if encoder_out.get("lm_cumsum") is not None:
            encoder_states["lm_cumsum"] = encoder_out["lm_cumsum"].index_select(
                0, new_order
            )
        if lm_out is not None:
            if lm_out["lm_input"] is not None:
                lm_out["lm_input"] = lm_out["lm_input"].index_select(0, new_order)
            if lm_out["lm_out"] is not None:
                lm_out["lm_out"] = lm_out["lm_out"].index_select(1, new_order)
            if lm_out["lm_padding_mask"] is not None:
                lm_out["lm_padding_mask"] = lm_out["lm_padding_mask"].index_select(
                    0, new_order
                )

        return encoder_states, lm_out
