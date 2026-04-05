from typing import Optional

import os
import numpy as np
from numpy.typing import NDArray

try:
    from numba import config as numba_config
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    _NUMBA_AVAILABLE = False
    numba_config = None  # type: ignore[assignment]


def _validate_events(
    event_starts: NDArray[np.int64],
    event_ends: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.int64], int]:
    """
    Validate event span arrays.

    :param event_starts: Event start positions on the time grid.
    :param event_ends: Event end positions on the time grid.
    :returns: (validated_starts, validated_ends, n_timestamps)
    :raises ValueError: If input arrays are invalid.
    """
    starts = np.asarray(event_starts, dtype=np.int64)
    ends = np.asarray(event_ends, dtype=np.int64)

    if starts.ndim != 1 or ends.ndim != 1:
        raise ValueError("event_starts and event_ends must be 1-D arrays.")
    if len(starts) != len(ends):
        raise ValueError("event_starts and event_ends must have the same length.")
    if len(starts) == 0:
        raise ValueError("At least one event is required.")
    if np.any(starts < 0) or np.any(ends < 0):
        raise ValueError("Event positions must be non-negative.")
    if np.any(ends < starts):
        raise ValueError("Each event_end must be greater than or equal to event_start.")

    n_timestamps = int(max(starts.max(), ends.max()) + 1)
    return starts, ends, n_timestamps


def _splitmix64_next(state: np.uint64) -> tuple[np.uint64, np.uint64]:
    """
    SplitMix64 step.

    :param state: RNG state.
    :returns: (new_state, random_u64)
    """
    with np.errstate(over='ignore'):
        state = np.uint64(state + np.uint64(0x9E3779B97F4A7C15))
        z = state
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return state, z


def _u01_from_u64(u: np.uint64) -> float:
    """
    Convert u64 -> uniform(0,1) double (53-bit mantissa).
    """
    return float((u >> np.uint64(11)) * np.float64(1.0 / (1 << 53)))


def _sequential_bootstrap_indices_py(
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    sample_length: int,
    random_state: Optional[int],
) -> NDArray[np.int64]:
    n_events = int(len(starts))
    n_timestamps = int(max(starts.max(), ends.max()) + 1)

    concurrency = np.zeros(n_timestamps, dtype=np.int32)
    sampled = np.empty(sample_length, dtype=np.int64)
    candidate_u = np.empty(n_events, dtype=np.float64)

    state = np.uint64(0 if random_state is None else int(random_state) & 0xFFFFFFFFFFFFFFFF)

    for k in range(sample_length):
        for i in range(n_events):
            start_idx = int(starts[i])
            end_idx = int(ends[i])
            acc = 0.0
            count = end_idx - start_idx + 1
            for t in range(start_idx, end_idx + 1):
                acc += 1.0 / (float(concurrency[t]) + 1.0)
            candidate_u[i] = acc / float(count)

        prob_sum = float(candidate_u.sum())
        if not np.isfinite(prob_sum) or prob_sum <= 0.0:
            state, rnd = _splitmix64_next(state)
            chosen_idx = int(_u01_from_u64(rnd) * n_events)
            if chosen_idx >= n_events:
                chosen_idx = n_events - 1
        else:
            state, rnd = _splitmix64_next(state)
            r = _u01_from_u64(rnd) * prob_sum
            c = 0.0
            chosen_idx = 0
            for i in range(n_events):
                c += float(candidate_u[i])
                if r <= c:
                    chosen_idx = i
                    break

        sampled[k] = chosen_idx
        start_idx = int(starts[chosen_idx])
        end_idx = int(ends[chosen_idx])
        concurrency[start_idx:end_idx + 1] += 1

    return sampled


def _avg_uniqueness_of_sample_py(
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    sampled: NDArray[np.int64],
) -> float:
    n_timestamps = int(max(starts.max(), ends.max()) + 1)
    concurrency = np.zeros(n_timestamps, dtype=np.int32)

    for idx in sampled:
        idx_i = int(idx)
        start_idx = int(starts[idx_i])
        end_idx = int(ends[idx_i])
        concurrency[start_idx:end_idx + 1] += 1

    uniq_sum = 0.0
    for idx in sampled:
        idx_i = int(idx)
        start_idx = int(starts[idx_i])
        end_idx = int(ends[idx_i])
        acc = 0.0
        count = end_idx - start_idx + 1
        for t in range(start_idx, end_idx + 1):
            acc += 1.0 / float(concurrency[t])
        uniq_sum += acc / float(count)

    return float(uniq_sum / float(len(sampled)))


if _NUMBA_AVAILABLE:

    @njit(nogil=True)
    def _splitmix64_next_nb(state):
        state = np.uint64(state + np.uint64(0x9E3779B97F4A7C15))
        z = state
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return state, z


    @njit(nogil=True)
    def _u01_from_u64_nb(u):
        return float((u >> np.uint64(11)) * np.float64(1.0 / (1 << 53)))


    @njit(nogil=True)
    def _sequential_bootstrap_indices_nb(starts, ends, sample_length, seed_u64):
        n_events = len(starts)
        n_timestamps = int(max(starts.max(), ends.max()) + 1)

        concurrency = np.zeros(n_timestamps, dtype=np.int32)
        sampled = np.empty(sample_length, dtype=np.int64)
        candidate_u = np.empty(n_events, dtype=np.float64)

        state = np.uint64(seed_u64)

        for k in range(sample_length):
            for i in range(n_events):
                start_idx = int(starts[i])
                end_idx = int(ends[i])
                acc = 0.0
                count = end_idx - start_idx + 1
                for t in range(start_idx, end_idx + 1):
                    acc += 1.0 / (float(concurrency[t]) + 1.0)
                candidate_u[i] = acc / float(count)

            prob_sum = 0.0
            for i in range(n_events):
                prob_sum += float(candidate_u[i])

            if not np.isfinite(prob_sum) or prob_sum <= 0.0:
                state, rnd = _splitmix64_next_nb(state)
                chosen_idx = int(_u01_from_u64_nb(rnd) * n_events)
                if chosen_idx >= n_events:
                    chosen_idx = n_events - 1
            else:
                state, rnd = _splitmix64_next_nb(state)
                r = _u01_from_u64_nb(rnd) * prob_sum
                c = 0.0
                chosen_idx = 0
                for i in range(n_events):
                    c += float(candidate_u[i])
                    if r <= c:
                        chosen_idx = i
                        break

            sampled[k] = chosen_idx
            start_idx = int(starts[chosen_idx])
            end_idx = int(ends[chosen_idx])
            for t in range(start_idx, end_idx + 1):
                concurrency[t] += 1

        return sampled


    @njit(nogil=True)
    def _avg_uniqueness_of_sample_nb(starts, ends, sampled):
        n_timestamps = int(max(starts.max(), ends.max()) + 1)
        concurrency = np.zeros(n_timestamps, dtype=np.int32)

        for k in range(len(sampled)):
            idx = int(sampled[k])
            start_idx = int(starts[idx])
            end_idx = int(ends[idx])
            for t in range(start_idx, end_idx + 1):
                concurrency[t] += 1

        uniq_sum = 0.0
        for k in range(len(sampled)):
            idx = int(sampled[k])
            start_idx = int(starts[idx])
            end_idx = int(ends[idx])
            acc = 0.0
            count = end_idx - start_idx + 1
            for t in range(start_idx, end_idx + 1):
                acc += 1.0 / float(concurrency[t])
            uniq_sum += acc / float(count)

        return float(uniq_sum / float(len(sampled)))


def _use_numba_jit() -> bool:
    """
    Decide whether to use Numba JIT implementation.

    CI and most tests run with NUMBA_DISABLE_JIT=1, so we explicitly fall back to
    pure-Python to avoid warning noise and keep behavior predictable.
    """
    if not _NUMBA_AVAILABLE:
        return False
    if numba_config is not None and bool(numba_config.DISABLE_JIT):
        return False
    if os.environ.get("NUMBA_DISABLE_JIT") in {"1", "true", "True", "YES", "yes"}:
        return False
    return True


def sequential_bootstrap_indices(
    event_starts: NDArray[np.int64],
    event_ends: NDArray[np.int64],
    sample_length: int,
    random_state: Optional[int] = None
) -> NDArray[np.int64]:
    """
    Draw bootstrap indices using AFML-style sequential bootstrapping.

    At each draw, candidate selection probabilities are proportional to
    the candidate's incremental average uniqueness under the current
    in-bag concurrency state.

    :param event_starts: Event start positions on the time grid.
    :param event_ends: Event end positions on the time grid.
    :param sample_length: Number of samples to draw (with replacement).
    :param random_state: Optional RNG seed.
    :returns: Drawn event indices (length = sample_length).
    :raises ValueError: If inputs are invalid.
    """
    starts, ends, _ = _validate_events(event_starts, event_ends)

    if sample_length <= 0:
        raise ValueError("sample_length must be > 0.")

    if _use_numba_jit():
        seed_u64 = np.uint64(0 if random_state is None else int(random_state) & 0xFFFFFFFFFFFFFFFF)
        return _sequential_bootstrap_indices_nb(starts, ends, int(sample_length), seed_u64)

    return _sequential_bootstrap_indices_py(starts, ends, int(sample_length), random_state)


def avg_uniqueness_of_sample(
    event_starts: NDArray[np.int64],
    event_ends: NDArray[np.int64],
    sampled_indices: NDArray[np.int64]
) -> float:
    """
    Compute average uniqueness of sampled events under in-bag concurrency.

    :param event_starts: Event start positions on the time grid.
    :param event_ends: Event end positions on the time grid.
    :param sampled_indices: In-bag sampled event indices.
    :returns: Mean uniqueness over sampled observations.
    :raises ValueError: If inputs are invalid.
    """
    starts, ends, _ = _validate_events(event_starts, event_ends)

    sampled = np.asarray(sampled_indices, dtype=np.int64)
    if sampled.ndim != 1:
        raise ValueError("sampled_indices must be a 1-D array.")
    if len(sampled) == 0:
        raise ValueError("sampled_indices must not be empty.")
    if np.any(sampled < 0) or np.any(sampled >= len(starts)):
        raise ValueError("sampled_indices contains out-of-range event id.")

    if _use_numba_jit():
        return float(_avg_uniqueness_of_sample_nb(starts, ends, sampled))

    return _avg_uniqueness_of_sample_py(starts, ends, sampled)
