import math
import uuid
from typing import Optional, Dict, Tuple, OrderedDict

import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.nn import LayerNorm as ESM1bLayerNorm
from mindspore.common.initializer import initializer, XavierUniform, XavierNormal, Constant


def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + ops.erf(x / math.sqrt(2.0)))


def utils_softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return ops.softmax(x.float(), axis=dim)
    else:
        return ops.softmax(x, axis=dim)


class FairseqIncrementalState(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: str) -> str:
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(
            self,
            incremental_state: Optional[Dict[str, Dict[str, Optional[ms.Tensor]]]],
            key: str,
    ) -> Optional[Dict[str, Optional[ms.Tensor]]]:
        """Helper for getting incremental state for an nn.Module."""
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
            self,
            incremental_state: Optional[Dict[str, Dict[str, Optional[ms.Tensor]]]],
            key: str,
            value: Dict[str, Optional[ms.Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[ms.Tensor]]]]:
        """Helper for setting incremental state for an nn.Module."""
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state


def with_incremental_state(cls):
    cls.__bases__ = (FairseqIncrementalState,) + tuple(
        b for b in cls.__bases__ if b != FairseqIncrementalState
    )
    return cls


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx = padding_idx)
        self.max_positions = num_embeddings
        self.embedding = nn.Embedding(vocab_size = num_embeddings, embedding_size = embedding_dim, padding_idx = self.padding_idx)

    def construct(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        if ops.shape(input)[1] > self.max_positions:
            raise ValueError(
                f"Sequence length {input.size(1)} above maximum "
                f" sequence length of {self.max_positions}"
            )
        mask = input.ne(self.padding_idx).int()
        positions = (ms.Tensor(ops.cumsum(mask, axis=1),dtype = mask.dtype) * mask).long() + self.padding_idx

        return self.embedding(positions)

class TransformerLayer(nn.Cell):
    """Transformer layer block."""

    def __init__(
            self,
            embed_dim,
            ffn_embed_dim,
            attention_heads,
            add_bias_kv=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self._init_submodules(add_bias_kv)

    def _init_submodules(self, add_bias_kv):
        super().__init__(nn.Cell)
        BertLayerNorm = ESM1bLayerNorm

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
        )
        self.self_attn_layer_norm = BertLayerNorm([self.embed_dim])

        self.fc1 = nn.Dense(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Dense(self.ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = BertLayerNorm([self.embed_dim])
        self.layer_gated = nn.Dense(self.embed_dim, 1, bias_init=False)

    def construct(
        self, x, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False, with_prompt_num=0
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn.construct(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            attn_mask=self_attn_mask,
        )
        if with_prompt_num != 0:
            l2_normalize = ops.L2Normalize(axis=2)
            gate = self.layer_gated(l2_normalize(x[-with_prompt_num:, :, :].copy())).mean()
            x[:-with_prompt_num, :, :] = (1 - gate) * x[:-with_prompt_num, :, :].copy()
        x = residual + x
        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn


class ProteinBertModel(nn.Cell):
    """
        RoBERTa Large architecture for encoding protein sequence.

        Args:
            layer_num (int, default=33): number of layers
            embed_dim (int, default=1280): embedding dimension
            ffn_embed_dim (int, default=5120): embedding dimension for feedforward network
            attention_head_num (int, default=20): number of attention heads
            max_position_num (int, default=1024): number of positional embeddings to learn
            emb_layer_norm_before (bool, default=True): whether apply layer norm before multi-head attentions
            checkpoint_path (str): the path of pre-trained checkpoint
    """
    def __init__(self, args, alphabet) -> None:
        super().__init__()
        self.args = args
        self.max_position_num = args.max_positions
        self.layer_num = args.num_layers
        self.attention_head_num = args.attention_heads
        self.embed_dim = args.embed_dim
        self.ffn_embed_dim = args.ffn_embed_dim
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.emb_layer_norm_before = getattr(self.args, "emb_layer_norm_before", False)

        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.embed_dim, padding_idx=self.padding_idx
        )
        self.layers = nn.CellList(
            [
                TransformerLayer(
                    self.embed_dim,
                    self.ffn_embed_dim,
                    self.attention_head_num,
                    add_bias_kv=False,
                )
                for _ in range(self.layer_num)
            ]
        )
        self.embed_scale = 1
        self.embed_positions = LearnedPositionalEmbedding(
            self.max_position_num, self.embed_dim, self.padding_idx
        )
        self.emb_layer_norm_before = (
            ESM1bLayerNorm([self.embed_dim]) if self.emb_layer_norm_before else None
        )
        self.emb_layer_norm_after = ESM1bLayerNorm([self.embed_dim])

    def construct(self, tokens, attn_mask=None, repr_layers=[], need_head_weights=False, with_prompt_num=0,
                learnable_prompt=None):
        global layer_idx, attn_weights
        assert tokens.ndim == 2
        #padding_mask = tokens.eq(self.padding_idx)  # B, T
        padding_mask = ops.equal(tokens, self.padding_idx)
        if with_prompt_num != 0 or learnable_prompt != None:
            x = self.embed_scale * self.embed_tokens(tokens)
            if with_prompt_num != 0:
                x[:, :-with_prompt_num, :] = x[:, :-with_prompt_num, :] + self.embed_positions(
                    tokens[:, :-with_prompt_num])
            else:
                x = x + self.embed_positions(tokens)
            if learnable_prompt is not None:
                learned_prompt_num = learnable_prompt.size(0)
                learnable_prompt = learnable_prompt.repeat(x.size(0), 1, 1)
                x = ops.cat([x, learnable_prompt], 1)
                padding_mask = ops.cat(
                    [padding_mask, ops.zeros((x.size(0), learned_prompt_num), dtype=padding_mask.dtype)], axis=1)
                with_prompt_num += learned_prompt_num
        else:
            x = self.embed_scale * self.embed_tokens(tokens)
            if getattr(self.args, 'token_dropout', True):
                x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
                # x: B x T x C
                mask_ratio_train = 0.15 * 0.8
                src_lengths = (~padding_mask).sum(-1)
                mask_ratio_observed = (tokens == self.mask_idx).sum(-1).float() / src_lengths
                x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
                x = x.to(next(self.embed_tokens.parameters()).dtype)
            x = x + self.embed_positions(tokens)
        attention_mask = ops.repeat_interleave(ms.Tensor(padding_mask,dtype = ms.dtype.int64), ops.shape(padding_mask)[1] * 20, axis=0).reshape(-1,
                                                                                                  ops.shape(padding_mask)[1],
                                                                                                  ops.shape(padding_mask)[1])
        attention_mask = ms.Tensor(attention_mask, dtype = ms.dtype.bool_)
        if with_prompt_num != 0:
            attention_mask[:, -with_prompt_num:, :-with_prompt_num] = True
        if self.emb_layer_norm_before:
            x = self.emb_layer_norm_before(x)
        if padding_mask is not None:
            #x = x * (1 - ms.Tensor(padding_mask.unsqueeze(-1),ms.dtype.int64))
            x = x * (1 - padding_mask.unsqueeze(-1))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x
        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        #x = x.transpose(0, 1)
        x = ops.swapaxes(x, 0, 1)
        if not padding_mask.any():
            padding_mask = None
        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x, self_attn_mask=attention_mask, self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
                with_prompt_num=with_prompt_num
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = ops.swapaxes(x, 0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(ops.swapaxes(attn, 1, 0))

        x = self.emb_layer_norm_after(x)
        x = ops.swapaxes(x, 0, 1)  # (T, B, E) => (B, T, E)
        # last hidden representation should have layer norm applied
        if (layer_idx + 2) in repr_layers:
            hidden_representations[layer_idx + 2] = x

        if with_prompt_num != 0:
            x = x[:, :-with_prompt_num, :]

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = ops.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
        return result

    # def load(self, args, alphabet):
    #     model = ProteinBertModel(args, alphabet)
    #     with torch.no_grad():
    #         new_state_dict = OrderedDict()
    #         with open(args.checkpoint_path, 'rb') as f:
    #             pretrain_decoder_dict = torch.load(f, map_location=torch.device('cpu'))
    #             for k, v in pretrain_decoder_dict['model'].items():
    #                 if 'sentence_encoder.' in k:
    #                     k = k.replace('encoder.sentence_encoder.', '')
    #                     new_state_dict[k] = v
    #         model.load_state_dict(new_state_dict)

        # return model