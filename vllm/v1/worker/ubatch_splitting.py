# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Sequence

import torch

from vllm.config import VllmConfig
from vllm.forward_context import DPMetadata
from vllm.logger import init_logger
from vllm.utils import round_up
from vllm.v1.worker.ubatch_utils import (UBatchSlice, UBatchSlices,
                                         is_second_ubatch_empty)
from vllm.v1.worker.ubatch_utils import UBatchTwoChunk
from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


def should_ubatch_with_num_tokens(
    should_ubatch: bool,
    orig_num_tokens_per_ubatch: int,
    padded_num_tokens_per_ubatch: int,
    vllm_config: VllmConfig,
) -> tuple[bool, Optional[torch.Tensor]]:
    dp_size = vllm_config.parallel_config.data_parallel_size
    dp_rank = vllm_config.parallel_config.data_parallel_rank
    return DPMetadata.should_ubatch_across_dp(should_ubatch,
                                              orig_num_tokens_per_ubatch,
                                              padded_num_tokens_per_ubatch,
                                              dp_size, dp_rank)


def get_dp_padding_ubatch(
        num_tokens_unpadded: int, num_tokens_padded: int,
        should_attempt_ubatching: bool,
        vllm_config: VllmConfig) -> tuple[bool, Optional[torch.Tensor]]:
    """
    1. Decides if each DP rank is going to microbatch. Either all ranks
    run with microbatching or none of them do. If this function decides
    not to run with microbatching. It will "abort" meaning that no padding
    information will be returned to the caller. It will return (False, 0, None)

    2. Determines the total number of tokens that each rank will run.
    All ranks will be padded out so that the run with the same number
    of tokens

    Returns: tuple[
        should_ubatch: Are all DP ranks going to microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including padding. Will be
        None if should_ubatch if False
    ]

    """
    assert num_tokens_padded >= num_tokens_unpadded
    dp_size = vllm_config.parallel_config.data_parallel_size
    #if dp_size == 1:
    #    # Early exit.
    #    return False, None

    # If this DP rank doesn't want to attempt microbatching
    if not should_attempt_ubatching:
        (should_ubatch, num_tokens_across_dp) = should_ubatch_with_num_tokens(
            False, 0, 0, vllm_config)
        logger.error(f"cwndmiao debug, get_dp_padding_ubatch 1, should_ubatch: {should_ubatch}, num_tokens_across_dp: {num_tokens_across_dp}")
        assert should_ubatch is False
        assert num_tokens_across_dp is None
        return should_ubatch, num_tokens_across_dp

    # Round up to the next multiple of two for even divisibility
    num_tokens_padded = round_up(num_tokens_padded, 2)
    num_tokens_per_ubatch = num_tokens_padded // 2
    should_ubatch = True
    logger.error(f"cwndmiao debug, get_dp_padding_ubatch 2, should_ubatch: {should_ubatch}, num_tokens_per_ubatch: {num_tokens_per_ubatch}, num_tokens_padded: {num_tokens_padded}, num_tokens_unpadded: {num_tokens_unpadded}")

    # Sanity Check that the existing padding isn't giving us an empty second
    # ubatch. Abort if so
    if is_second_ubatch_empty(num_tokens_unpadded, num_tokens_padded):
        logger.debug("Aborting ubatching %s %s", num_tokens_unpadded,
                     num_tokens_padded)
        should_ubatch = False

    # Note that we compute the number of padded tokens per ubatch
    (should_ubatch, num_tokens_across_dp) = should_ubatch_with_num_tokens(
        should_ubatch, num_tokens_unpadded // 2, num_tokens_per_ubatch,
        vllm_config)
    if not should_ubatch:
        assert num_tokens_across_dp is None
        return should_ubatch, num_tokens_across_dp

    assert num_tokens_across_dp is not None

    max_tokens_across_dp_cpu = int(torch.max(num_tokens_across_dp).item())
    num_tokens_after_padding = torch.tensor([max_tokens_across_dp_cpu] *
                                            dp_size,
                                            device="cpu",
                                            dtype=torch.int32)
    return should_ubatch, num_tokens_after_padding


def ubatch_split(
    #max_num_scheduled_tokens: int,
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    vllm_config: VllmConfig,
    scheduler_output: SchedulerOutput,
    tokens: Sequence[int],
) -> tuple[Optional[UBatchSlices], Optional[torch.Tensor]]:
    """
    Coordinates amongst all DP ranks to determine if and how the full batch
    should be split into microbatches.

    Returns: tuple[
        ubatch_slices: if this is set then all DP ranks have agreed to 
        microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including padding. Will be
        None if ubatch_slices is None
    ]

    """
    parallel_config = vllm_config.parallel_config
    # Don't bother with the should_ubatch handshaking unless microbatching
    # is enabled
    if not parallel_config.enable_microbatching:
        return (None, None)

    # # Check preconditions for microbatching
    # should_attempt_ubatching = \
    #     parallel_config.enable_microbatching and \
    #     num_tokens_unpadded >= \
    #     parallel_config.microbatching_token_threshold \
    #     and max_num_scheduled_tokens == 1
    # logger.error(f"cwndmiao debug, ubatch_split, parallel_config.enable_microbatching: {parallel_config.enable_microbatching}, "
    #              f"num_tokens_unpadded: {num_tokens_unpadded} >= {parallel_config.microbatching_token_threshold}, "
    #              f"num_tokens_padded: {num_tokens_padded}, "
    #              f"max_num_scheduled_tokens: {max_num_scheduled_tokens}, "
    #              f"should_attempt_ubatching: {should_attempt_ubatching}")
    # # TODO(miaotianxiang):
    # should_attempt_ubatching = True

    # # Don't microbatch unless every other DP worker is also microbatching
    # num_tokens_after_padding = None
    # (should_ubatch, num_tokens_after_padding) = get_dp_padding_ubatch(
    #     num_tokens_unpadded, num_tokens_padded, should_attempt_ubatching,
    #     vllm_config)
    # logger.error(f"cwndmiao debug, ubatch_split, should_ubatch: {should_ubatch}, num_tokens_after_padding: {num_tokens_after_padding}")
    # if not should_ubatch:
    #     return (None, None)

    # # This doesn't actually pad the ubatch slices. It just initializes the
    # # split point to the padded value so that padding can be applied
    # # to the second ubatch in pad_out_ubatch_slice after attention
    # # metadata creation
    # assert num_tokens_after_padding is not None
    # total_num_tokens_per_ubatch = int(num_tokens_after_padding[0].item())
    # padded_first_ubatch_slice = slice(0, total_num_tokens_per_ubatch)
    # padded_second_ubatch_slice = slice(total_num_tokens_per_ubatch,
    #                                    num_tokens_unpadded)

    # # Note there's an assumption here that there's 1 token per request
    # ubatch_slices = [
    #     UBatchSlice(padded_first_ubatch_slice, padded_first_ubatch_slice),
    #     UBatchSlice(padded_second_ubatch_slice, padded_second_ubatch_slice)
    # ]

    # return (ubatch_slices, num_tokens_after_padding)

    # Don't microbatch unless every other DP worker is also microbatching
    should_attempt_ubatching = True
    num_tokens_after_padding = None
    (should_ubatch, num_tokens_after_padding) = get_dp_padding_ubatch(
        num_tokens_unpadded, num_tokens_padded, should_attempt_ubatching,
        vllm_config)
    logger.error(f"cwndmiao debug, ubatch_split 1, should_ubatch: {should_ubatch}, num_tokens_after_padding: {num_tokens_after_padding}, "
                 f"num_tokens_unpadded: {num_tokens_unpadded}, num_tokens_padded: {num_tokens_padded}")
    if not should_ubatch:
        return (None, None)

    # TODO(miaotianxiang): 搬迁自sglang
    #token_num_per_seq = get_token_num_per_seq(
    #        forward_mode=batch.forward_mode, spec_info=batch.spec_info
    #)

    scheduler_output.is_two_chunk_split = _is_two_chunk_split_enabled(tokens)
    scheduler_output.tbo_split_seq_index = compute_split_seq_index(
        #forward_mode=batch.forward_mode,
        #num_tokens=num_tokens,
        #extend_lens=None,
        #token_num_per_seq=token_num_per_seq,
        tokens
    )
    # For simplicity, when two_batch_overlap is enabled, we only capture CUDA Graph for tbo=true
    assert scheduler_output.tbo_split_seq_index is not None, f"{tokens=}"
    scheduler_output.tbo_split_token_index = compute_split_token_index(
        split_seq_index=scheduler_output.tbo_split_seq_index,
        #forward_mode=batch.forward_mode,
        #extend_seq_lens=None,
        #token_num_per_seq=token_num_per_seq,
        extend_seq_lens=tokens
    )
    logger.error(f"cwndmiao debug, ubatch_split 2, {tokens=}, {scheduler_output.tbo_split_seq_index=}, {scheduler_output.tbo_split_token_index=}")

    _tbo_children_num_token_non_padded = torch.zeros((2,), dtype=torch.int32)
    _tbo_children_num_token_non_padded[...] = (
        TboForwardBatchPreparer.compute_tbo_children_num_token_non_padded(scheduler_output, tokens)
    )
    logger.error(f"cwndmiao debug, ubatch_split 3, _tbo_children_num_token_non_padded: {_tbo_children_num_token_non_padded}")

    #TboForwardBatchPreparer.prepare_raw(
    #    scheduler_output,
    #    tbo_children_num_token_non_padded=_tbo_children_num_token_non_padded,
    #)

    ubatch_slices = [
        UBatchSlice(slice(0, scheduler_output.tbo_split_seq_index + 1 if scheduler_output.is_two_chunk_split else scheduler_output.tbo_split_seq_index),
                    slice(0, scheduler_output.tbo_split_token_index),
                    UBatchTwoChunk.FIRST_CHUNK_SPLIT if scheduler_output.is_two_chunk_split else UBatchTwoChunk.NO_CHUNK_SPLIT,
                    scheduler_output.tbo_split_token_index - sum(tokens[:scheduler_output.tbo_split_seq_index]) if scheduler_output.is_two_chunk_split else 0,
                    tokens[scheduler_output.tbo_split_seq_index] if scheduler_output.is_two_chunk_split else 0),
        UBatchSlice(slice(scheduler_output.tbo_split_seq_index, len(tokens)),
                    slice(scheduler_output.tbo_split_token_index, num_tokens_unpadded),
                    UBatchTwoChunk.SECOND_CHUNK_SPLIT if scheduler_output.is_two_chunk_split else UBatchTwoChunk.NO_CHUNK_SPLIT,
                    sum(tokens[:scheduler_output.tbo_split_seq_index + 1]) - scheduler_output.tbo_split_token_index if scheduler_output.is_two_chunk_split else 0,
                    tokens[scheduler_output.tbo_split_seq_index] if scheduler_output.is_two_chunk_split else 0)
    ]
    logger.error(f"cwndmiao debug, ubatch_split 4, ubatch_slices= {ubatch_slices}")

    return (ubatch_slices, num_tokens_after_padding)

class TboForwardBatchPreparer:
    @classmethod
    def compute_tbo_children_num_token_non_padded(cls, scheduler_output: SchedulerOutput, tokens: Sequence[int]):
        return cls.compute_tbo_children_num_token_non_padded_raw(
            tbo_split_token_index=scheduler_output.tbo_split_token_index,
            num_token_non_padded=scheduler_output.total_num_scheduled_tokens,
        )

    @classmethod
    def compute_tbo_children_num_token_non_padded_raw(
        cls, tbo_split_token_index: int, num_token_non_padded: int
    ):
        # TODO we may make padding on both sub-batches to make it slightly more balanced
        value_a = min(tbo_split_token_index, num_token_non_padded)
        value_b = max(0, num_token_non_padded - tbo_split_token_index)
        #return torch.tensor([value_a, value_b], dtype=torch.int32).to(
        #    device=global_server_args_dict["device"], non_blocking=True
        #)
        return torch.tensor([value_a, value_b], dtype=torch.int32)

    #@classmethod
    #def _compute_split_token_index(cls, scheduler_output: SchedulerOutput, tokens: Sequence[int]):
    #    #token_num_per_seq = get_token_num_per_seq(
    #    #    forward_mode=batch.forward_mode, spec_info=batch.spec_info
    #    #)
    #    return compute_split_token_index(
    #        split_seq_index=scheduler_output.tbo_split_seq_index,
    #        #forward_mode=batch.forward_mode,
    #        extend_seq_lens=tokens,
    #        #token_num_per_seq=token_num_per_seq,
    #    )

# TODO: may smartly disable TBO when batch size is too small b/c it will slow down
def compute_split_seq_index(
    #forward_mode: "ForwardMode",
    #num_tokens: int,
    #extend_lens: Optional[Sequence[int]],
    #token_num_per_seq: Optional[int],
    extend_lens: Sequence[int],
) -> Optional[int]:
    #if forward_mode == ForwardMode.EXTEND:
    #    assert extend_lens is not None
    #    return _split_extend_seqs(extend_lens)
    #elif forward_mode.is_target_verify() or forward_mode.is_decode():
    #    assert token_num_per_seq is not None
    #    return (num_tokens // token_num_per_seq) // 2
    #elif forward_mode.is_idle():
    #    assert num_tokens == 0
    #    return 0
    #else:
    #    raise NotImplementedError()
    assert extend_lens is not None
    return _split_extend_seqs(extend_lens)

def _split_extend_seqs(arr: Sequence[int]) -> int:
    if _is_two_chunk_split_enabled(arr):
        return _split_array_by_cum_less_than_half(arr)

    return _split_array_by_balanced_sum(arr)

def _is_two_chunk_split_enabled(extend_lens: Sequence[int]) -> bool:
    if extend_lens is None:
        return False

    vanilla_split_seq_index = _split_array_by_balanced_sum(extend_lens)
    left_sum = sum(extend_lens[:vanilla_split_seq_index])
    overall_sum = sum(extend_lens)
    #threshold = get_tbo_token_distribution_threshold()
    threshold = 0.48
    assert threshold <= 0.5, f"{threshold=}"
    return left_sum < overall_sum * threshold or left_sum > overall_sum * (
        1 - threshold
    )

def _split_array_by_cum_less_than_half(arr: Sequence[int]) -> int:
    left_sum = 0
    overall_sum = sum(arr)
    half_sum = overall_sum // 2
    chosen_index = 0

    for i in range(len(arr)):
        left_sum += arr[i]
        if left_sum > half_sum:
            chosen_index = i
            break

    return chosen_index

def _split_array_by_balanced_sum(arr: Sequence[int]) -> int:
    overall_sum = sum(arr)
    left_sum = 0
    min_diff = float("inf")
    best_index = 0

    for i in range(1, len(arr)):
        left_sum += arr[i - 1]
        right_sum = overall_sum - left_sum
        diff = abs(left_sum - right_sum)
        if diff <= min_diff:
            min_diff = diff
            best_index = i
        else:
            break

    return best_index

def compute_split_token_index(
    split_seq_index: int,
    #forward_mode: "ForwardMode",
    #extend_seq_lens: Optional[Sequence[int]],
    #token_num_per_seq: Optional[int],
    extend_seq_lens: Sequence[int],
) -> int:
    #if forward_mode == ForwardMode.EXTEND:
    #    assert extend_seq_lens is not None
    #    if _is_two_chunk_split_enabled(extend_seq_lens):
    #        return sum(extend_seq_lens) // 2
    #    return sum(extend_seq_lens[:split_seq_index])
    #elif forward_mode.is_target_verify() or forward_mode.is_decode():
    #    assert token_num_per_seq is not None
    #    return split_seq_index * token_num_per_seq
    #elif forward_mode.is_idle():
    #    assert split_seq_index == 0
    #    return 0
    #else:
    #    raise NotImplementedError
    assert extend_seq_lens is not None
    if _is_two_chunk_split_enabled(extend_seq_lens):
        return sum(extend_seq_lens) // 2
    return sum(extend_seq_lens[:split_seq_index])