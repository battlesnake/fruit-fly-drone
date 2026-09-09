from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from flydrone.connectome_data import farthest_point_sample, reverse_bfs_next, trace_path


def test_farthest_point_sample_is_deterministic_and_spread() -> None:
    points = np.asarray([(0, 0), (0, 1), (1, 0), (1, 1), (0.5, 0.5)], dtype=np.float32)
    first = farthest_point_sample(points, 4)
    second = farthest_point_sample(points, 4)

    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 4
    assert 4 not in first  # corners cover the square better than its centre


def test_reverse_bfs_reconstructs_directed_path() -> None:
    adjacency = csr_matrix(
        (np.ones(4), (np.asarray([0, 1, 2, 0]), np.asarray([1, 2, 3, 4]))),
        shape=(5, 5),
    )
    distance, next_hop = reverse_bfs_next(adjacency, np.asarray([3]))

    assert distance.tolist() == [3, 2, 1, 0, -1]
    assert trace_path(0, distance, next_hop, max_hops=3) == [0, 1, 2, 3]
    assert trace_path(4, distance, next_hop, max_hops=3) == []
