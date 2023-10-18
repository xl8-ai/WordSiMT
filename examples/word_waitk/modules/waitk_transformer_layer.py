"""
code from https://github.com/elbayadm/attn2d/blob/master/examples/waitk/modules/transformer_layers.py
"""
from numpy.random import uniform

from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.modules.transformer_layer import TransformerDecoderLayer


class WaitkTransformerDecoderLayer(TransformerDecoderLayer):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False):
        super().__init__(args, no_encoder_attn=no_encoder_attn)
        self.need_attn = False
        if args.language_model_name is not None:
            self.lm_attn = MultiheadAttention(
                self.embed_dim,
                args.encoder_attention_heads,
                kdim=args.lm_out_dim,
                vdim=args.lm_out_dim,
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
            )

            self.encoder_ratio = args.encoder_ratio
            self.lm_ratio = args.lm_ratio
            self.encoder_lm_dropout = getattr(args, "encoder_lm_dropout", False)
            self.encoder_lm_dropout_ratio = getattr(
                args, "encoder_lm_dropout_ratio", 0.25
            )
            assert (
                self.encoder_lm_dropout_ratio >= 0.0
                and self.encoder_lm_dropout_ratio <= 0.5
            )
            self.encoder_lm_mixup = getattr(args, "encoder_lm_mixup", False)
        else:
            self.lm_attn = None

    def forward(
        self,
        x,
        encoder_out,
        encoder_padding_mask,
        encoder_attn_mask=None,
        incremental_state=None,
        prev_self_attn_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        need_attn=False,
        static_encoder_attn_kv=False,
        lm_out=None,
        lm_padding_mask=None,
        lm_attn_mask=None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x_encoder_attn, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=(True if encoder_out is None else False)
                or static_encoder_attn_kv,
                attn_mask=encoder_attn_mask,
                need_weights=(not self.training and self.need_attn) or need_attn,
            )
            x_encoder_attn = self.dropout_module(x_encoder_attn)
            if self.lm_attn is not None:
                x_lm_attn, _ = self.lm_attn(
                    query=x,
                    key=lm_out,
                    value=lm_out,
                    key_padding_mask=lm_padding_mask,
                    incremental_state=incremental_state,
                    static_kv=(True if lm_out is None else False)
                    or static_encoder_attn_kv,
                    attn_mask=lm_attn_mask,
                )
                x_lm_attn = self.dropout_module(x_lm_attn)
                ratios = self.get_ratio()
                x = self.residual_connection(
                    ratios[0] * x_encoder_attn + ratios[1] * x_lm_attn, residual
                )
            else:
                x = self.residual_connection(x_encoder_attn, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, attn

    def get_ratio(self):
        if self.encoder_lm_dropout:
            frand = float(uniform(0, 1))
            if self.encoder_lm_mixup and self.training:
                return [frand, 1 - frand]
            if frand < self.encoder_lm_dropout_ratio and self.training:
                return [1, 0]
            elif frand > 1 - self.encoder_lm_dropout_ratio and self.training:
                return [0, 1]
            else:
                return [0.5, 0.5]
        else:
            return [self.encoder_ratio, self.lm_ratio]
