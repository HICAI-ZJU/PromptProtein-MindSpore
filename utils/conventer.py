from typing import Sequence, Tuple

import mindspore as ms
import mindspore.ops as ops
class PromptConverter(object):
    """ Convert Batch to Pre-train Task Needed Format
    input:    batch [(label1, sequence1), (label2, sequence2), ...]
    output:   labels, str_sequences, origin_tokens, masked_tokens(have been masked and padding), mask_ids
    """
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.pad_idx = alphabet.padding_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx

    def __call__(self, sequence, prompt_toks=[]):
        global encoded_prompt
        batch_size = 1
        if len(prompt_toks) != 0:
            encoded_prompt = ms.Tensor([self.alphabet.encode(prompt_tok)[0] for prompt_tok in prompt_toks])
        encoded_sequence = self.alphabet.encode(sequence)
        max_encoded_sequences_length = min(len(encoded_sequence), 1022)
        tokens = ms.numpy.empty(
            (
                batch_size,
                max_encoded_sequences_length + len(prompt_toks) + 2,
            ),
            dtype=ms.int64
        )
        shape = ops.Shape()
        size = shape(tokens)
        tokens = ops.fill(type=ms.int64, shape=size, value=self.pad_idx)

        sequence_length = min(len(encoded_sequence), 1022)
        encoded_sequence = ms.Tensor(encoded_sequence, dtype=ms.int64)
        tokens[0, 0] = self.cls_idx
        tokens[0, 1:sequence_length+1] = encoded_sequence[:sequence_length]
        tokens[0, sequence_length+1] = self.eos_idx

        if len(prompt_toks) != 0:
            tokens[0, -len(prompt_toks):] = encoded_prompt.copy()

        return encoded_sequence, tokens

