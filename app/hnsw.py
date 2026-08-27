"""
A simplified, from-scratch implementation of HNSW (Hierarchical Navigable Small
World) for approximate nearest-neighbor search over face embeddings.

This intentionally trades some of the original paper's sophistication (e.g. the
neighbor-diversity heuristic) for readability, while keeping the core ideas:
- a layered graph, sparser at the top, dense at the bottom
- greedy descent through layers to find a good entry point fast
- a widened best-first search at the bottom layer for the real candidate list
"""
import math
import random
import heapq
import pickle

import numpy as np


#def euclidean_distance(a, b):
#    return float(np.linalg.norm(a - b))

def euclidean_distance(a, b):   #cosine similarity
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(1 - similarity)

class HNSWIndex:
    def __init__(self, dim, M=16, ef_construction=200, ef_search=50,
                 distance_fn=euclidean_distance):
        self.dim = dim
        self.M = M                     # max neighbors per node, layers > 0
        self.M0 = M * 2                # max neighbors per node, layer 0 (denser)
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.distance_fn = distance_fn
        self.level_mult = 1.0 / math.log(M)

        self.vectors = {}      # node_id -> np.array
        self.neighbors = {}    # node_id -> {layer: set(neighbor_ids)}
        self.entry_point = None
        self.max_layer = -1
        self._next_id = 0

    # ---------- internal helpers ----------

    def _random_level(self):
        return int(-math.log(random.random()) * self.level_mult)

    def _distance(self, id_a, id_b):
        return self.distance_fn(self.vectors[id_a], self.vectors[id_b])

    def _distance_to_query(self, query, node_id):
        return self.distance_fn(query, self.vectors[node_id])

    def _search_layer(self, query, entry_points, ef, layer):
        """Best-first search restricted to a single layer.
        Returns up to `ef` (distance, node_id) pairs, sorted ascending by distance.
        """
        visited = set(entry_points)
        candidates = [(self._distance_to_query(query, ep), ep) for ep in entry_points]
        heapq.heapify(candidates)

        # results is a max-heap (via negated distance) capped at size `ef`
        results = [(-d, i) for d, i in candidates]
        heapq.heapify(results)

        while candidates:
            dist, current = heapq.heappop(candidates)

            if results and len(results) >= ef and dist > -results[0][0]:
                break  # nothing left in the frontier can improve our results

            for neighbor in self.neighbors.get(current, {}).get(layer, ()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                d = self._distance_to_query(query, neighbor)

                if len(results) < ef or d < -results[0][0]:
                    heapq.heappush(candidates, (d, neighbor))
                    heapq.heappush(results, (-d, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)

        return sorted((-d, i) for d, i in results)

    def _select_neighbors(self, candidates, m):
        """Simplified selection: keep the m closest candidates."""
        return [node_id for _, node_id in sorted(candidates)[:m]]

    # ---------- public API ----------

    def insert(self, vector):
        vector = np.asarray(vector, dtype=np.float32)
        new_id = self._next_id
        self._next_id += 1
        level = self._random_level()

        self.vectors[new_id] = vector
        self.neighbors[new_id] = {l: set() for l in range(level + 1)}

        if self.entry_point is None:
            self.entry_point = new_id
            self.max_layer = level
            return new_id

        current_nearest = [self.entry_point]

        # Descend greedily from the top layer down to level+1 (ef=1: just track the closest point)
        for layer in range(self.max_layer, level, -1):
            current_nearest = [node_id for _, node_id in
                                self._search_layer(vector, current_nearest, ef=1, layer=layer)]

        # From min(level, max_layer) down to 0: find real candidates and wire up connections
        for layer in range(min(level, self.max_layer), -1, -1):
            candidates = self._search_layer(vector, current_nearest, ef=self.ef_construction, layer=layer)
            max_conn = self.M0 if layer == 0 else self.M
            chosen = self._select_neighbors(candidates, max_conn)

            for neighbor_id in chosen:
                self.neighbors[new_id][layer].add(neighbor_id)
                self.neighbors[neighbor_id].setdefault(layer, set()).add(new_id)

                # prune the neighbor's connections at this layer if it's now over capacity
                if len(self.neighbors[neighbor_id][layer]) > max_conn:
                    reranked = sorted(
                        (self._distance(neighbor_id, nb), nb)
                        for nb in self.neighbors[neighbor_id][layer]
                    )
                    self.neighbors[neighbor_id][layer] = set(nb for _, nb in reranked[:max_conn])

            current_nearest = [node_id for _, node_id in candidates] or current_nearest

        if level > self.max_layer:
            self.max_layer = level
            self.entry_point = new_id

        return new_id

    def search(self, query, k=5):
        """Returns up to k (distance, node_id) pairs, sorted ascending by distance."""
        if self.entry_point is None:
            return []

        query = np.asarray(query, dtype=np.float32)
        current_nearest = [self.entry_point]

        for layer in range(self.max_layer, 0, -1):
            current_nearest = [node_id for _, node_id in
                                self._search_layer(query, current_nearest, ef=1, layer=layer)]

        ef = max(self.ef_search, k)
        results = self._search_layer(query, current_nearest, ef=ef, layer=0)
        return results[:k]

    def size(self):
        return len(self.vectors)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
