# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

from typing_extensions import TypeAlias

import enum

class UBatchTwoChunk(enum.Enum):
    NO_CHUNK_SPLIT = 0
    FIRST_CHUNK_SPLIT = 1
    SECOND_CHUNK_SPLIT = 2

@dataclass
class UBatchSlice:
    request_slice: slice
    token_slice: slice
    is_two_chunk_split: UBatchTwoChunk = UBatchTwoChunk.NO_CHUNK_SPLIT
    chunk_len: int = 0  # valid only when is_two_chunk_split != NO_CHUNK_SPLIT
    full_len: int = 0  # valid only when is_two_chunk_split != NO_CHUNK_SPLIT


UBatchSlices: TypeAlias = list[UBatchSlice]


def is_second_ubatch_empty(orig_num_tokens_per_ubatch: int,
                           padded_num_tokens_per_ubatch: int) -> bool:
    #return padded_num_tokens_per_ubatch >= 2 * orig_num_tokens_per_ubatch
    return False
