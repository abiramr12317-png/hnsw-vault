"""
Validates the custom HNSW implementation.

Two things matter here:
1. Correctness - does it actually find the true nearest neighbors (recall@K),
   compared against a brute-force ground truth?
2. Speed - is it actually faster than brute force as the dataset grows?

Run with:  pytest tests/test_hnsw.py -s
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.hnsw import HNSWIndex, euclidean_distance


def brute_force_knn(vectors, query, k):
    dists = [(euclidean_distance(query, v), i) for i, v in vectors.items()]
    return sorted(dists)[:k]


def build_random_index(n, dim, seed=42, **kwargs):
    rng = np.random.default_rng(seed)
    index = HNSWIndex(dim=dim, **kwargs)
    raw_vectors = {}
    for _ in range(n):
        vec = rng.random(dim).astype(np.float32)
        node_id = index.insert(vec)
        raw_vectors[node_id] = vec
    return index, raw_vectors


def test_recall_at_k():
    dim, n, k = 32, 500, 10
    index, raw_vectors = build_random_index(n, dim, M=16, ef_construction=200, ef_search=50)

    rng = np.random.default_rng(1)
    query_ids = rng.choice(list(raw_vectors.keys()), size=30, replace=False)

    total_recall = 0.0
    for qid in query_ids:
        query = raw_vectors[qid]
        true_neighbors = {i for _, i in brute_force_knn(raw_vectors, query, k)}
        hnsw_neighbors = {i for _, i in index.search(query, k=k)}
        total_recall += len(true_neighbors & hnsw_neighbors) / k

    avg_recall = total_recall / len(query_ids)
    print(f"\n[recall test] average recall@{k} over {len(query_ids)} queries: {avg_recall:.2%}")
    assert avg_recall > 0.7, "Recall too low - check neighbor selection / ef parameters"


def test_empty_index_search_returns_nothing():
    index = HNSWIndex(dim=8)
    assert index.search(np.random.rand(8), k=5) == []


def test_single_insert_and_query_returns_self():
    index = HNSWIndex(dim=8)
    vec = np.random.rand(8).astype(np.float32)
    node_id = index.insert(vec)
    results = index.search(vec, k=1)
    assert results[0][1] == node_id
    assert results[0][0] < 1e-5


def test_k_larger_than_dataset_does_not_crash():
    index, _ = build_random_index(n=5, dim=8)
    results = index.search(np.random.rand(8), k=100)
    assert len(results) <= 5


def test_duplicate_vectors():
    index = HNSWIndex(dim=8)
    vec = np.ones(8, dtype=np.float32)
    id_a = index.insert(vec)
    id_b = index.insert(vec)
    results = index.search(vec, k=2)
    found_ids = {r[1] for r in results}
    assert {id_a, id_b} <= found_ids


def test_speed_vs_brute_force():
    dim, k = 32, 10
    sizes = [200, 1000, 3000]

    print("\n[speed test] HNSW vs brute-force query latency")
    print(f"{'n':>6} | {'brute-force (ms)':>18} | {'hnsw (ms)':>10}")

    for n in sizes:
        index, raw_vectors = build_random_index(n, dim, M=16, ef_construction=200, ef_search=50)
        query = np.random.rand(dim).astype(np.float32)

        start = time.perf_counter()
        brute_force_knn(raw_vectors, query, k)
        brute_time = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        index.search(query, k=k)
        hnsw_time = (time.perf_counter() - start) * 1000

        print(f"{n:>6} | {brute_time:>18.3f} | {hnsw_time:>10.3f}")


if __name__ == "__main__":
    test_recall_at_k()
    test_empty_index_search_returns_nothing()
    test_single_insert_and_query_returns_self()
    test_k_larger_than_dataset_does_not_crash()
    test_duplicate_vectors()
    test_speed_vs_brute_force()
    print("\nAll tests passed.")
