from numpy.random import uniform

from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.modules.transformer_layer import TransformerEncoderLayer


class UnidirTransformerEncoderLayer(TransformerEncoderLayer):
    """Encoder layer block."""

    def __init__(self, args):
        super().__init__(args)
        if args.language_model_name is not None:
            self.lm_attn = MultiheadAttention(
                self.embed_dim,
                args.encoder_attention_heads,
                kdim=args.lm_out_dim,
                vdim=args.lm_out_dim,
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,  # This should be True to set static_kv=True
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
        encoder_padding_mask,
        self_attn_mask=None,
        incremental_state=None,
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
        x_self_attn, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            incremental_state=incremental_state,
            attn_mask=self_attn_mask,
        )
        x_self_attn = self.dropout_module(x_self_attn)
        if self.lm_attn is not None:
            x_lm_attn, _ = self.lm_attn(
                query=x,
                key=lm_out,
                value=lm_out,
                key_padding_mask=lm_padding_mask,
                incremental_state=incremental_state,
                static_kv=(True if lm_out is None else False),
                attn_mask=lm_attn_mask,
            )
            x_lm_attn = self.dropout_module(x_lm_attn)
            ratios = self.get_ratio()
            x = self.residual_connection(
                ratios[0] * x_self_attn + ratios[1] * x_lm_attn, residual
            )
        else:
            x = self.residual_connection(x_self_attn, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

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
        return x

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
