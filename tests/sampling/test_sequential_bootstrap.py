import numpy as np

from afmlkit.sampling import avg_uniqueness_of_sample, sequential_bootstrap_indices


def _make_overlapped_events():
    starts = np.arange(0, 40, dtype=np.int64)
    ends = np.minimum(starts + 10, 49).astype(np.int64)
    return starts, ends


def test_sequential_bootstrap_indices_is_reproducible():
    starts, ends = _make_overlapped_events()

    s1 = sequential_bootstrap_indices(starts, ends, sample_length=30, random_state=7)
    s2 = sequential_bootstrap_indices(starts, ends, sample_length=30, random_state=7)
    s3 = sequential_bootstrap_indices(starts, ends, sample_length=30, random_state=8)

    np.testing.assert_array_equal(s1, s2)
    assert not np.array_equal(s1, s3)


def test_sequential_bootstrap_indices_output_valid():
    starts, ends = _make_overlapped_events()
    sampled = sequential_bootstrap_indices(starts, ends, sample_length=80, random_state=42)

    assert len(sampled) == 80
    assert sampled.dtype == np.int64
    assert sampled.min() >= 0
    assert sampled.max() < len(starts)
    assert len(np.unique(sampled)) < len(sampled)


def test_sequential_bootstrap_improves_avg_uniqueness_against_uniform():
    starts, ends = _make_overlapped_events()
    sample_length = 40

    sb_scores = []
    uniform_scores = []
    for seed in range(20):
        sb_idx = sequential_bootstrap_indices(starts, ends, sample_length=sample_length, random_state=seed)
        sb_scores.append(avg_uniqueness_of_sample(starts, ends, sb_idx))

        rng = np.random.RandomState(seed)
        unif_idx = rng.choice(len(starts), size=sample_length, replace=True).astype(np.int64)
        uniform_scores.append(avg_uniqueness_of_sample(starts, ends, unif_idx))

    assert float(np.mean(sb_scores)) > float(np.mean(uniform_scores))
