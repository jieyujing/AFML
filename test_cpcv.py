import itertools
from scipy.special import comb
n_groups = 6
n_test_groups = 2
combos = list(itertools.combinations(range(n_groups), n_test_groups))
n_paths = int(comb(n_groups - 1, n_test_groups - 1))
paths = {p: {} for p in range(n_paths)}

for split_idx, combo in enumerate(combos):
    for group in combo:
        for p in range(n_paths):
            if group not in paths[p]:
                paths[p][group] = split_idx
                break

for p in paths:
    print(p, paths[p])
