import time
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx
import scipy.sparse
import scipy.sparse.csgraph


class LocalSearch:


    def __init__(
        self,
        problem,
        max_iterations: int = 10,   
        # multi-start restarts
        max_local_iterations: int = 300,  
        # neighbor trials per restart
        seed: Optional[int] = None,
        verbose: bool = False,
        # DP cap for beta<=1 to avoid O(n^2) inside many evaluations
        dp_lookahead_cap: int = 200,
        # chunking only when beta>1
        chunk_size: float = 500.0,
        time_limit_s: float = 15.0,
    ):
        self.problem = problem
        self.graph = problem.graph
        self.alpha = float(problem.alpha)
        self.beta = float(problem.beta)

        self.max_iterations = int(max_iterations)
        self.max_local_iterations = int(max_local_iterations)
        self.seed = seed
        self.verbose = bool(verbose)
        self.dp_lookahead_cap = int(dp_lookahead_cap)
        self.chunk_size = float(chunk_size)
        self.time_limit_s = float(time_limit_s)

        self.rng = random.Random(seed)

        # build virtual nodes 

        # For beta>1: optional chunking; for beta<=1: do not chunk
        real_cities = [n for n in self.graph.nodes if int(n) != 0]
        real_cities = [int(x) for x in real_cities]

        self.virtual_nodes: List[Dict[str, float]] = []
        self.city_to_v: Dict[int, List[int]] = {}

        if self.beta > 1.0:
            for city in real_cities:
                total_gold = float(self.graph.nodes[city]["gold"])
                if total_gold <= 0:
                    continue
                if total_gold > self.chunk_size:
                    rem = total_gold
                    while rem > 0:
                        amt = min(rem, self.chunk_size)
                        vidx = len(self.virtual_nodes)
                        self.virtual_nodes.append({"city": city, "gold": float(amt)})
                        self.city_to_v.setdefault(city, []).append(vidx)
                        rem -= amt
                else:
                    vidx = len(self.virtual_nodes)
                    self.virtual_nodes.append({"city": city, "gold": float(total_gold)})
                    self.city_to_v.setdefault(city, []).append(vidx)
        else:
            for city in real_cities:
                total_gold = float(self.graph.nodes[city]["gold"])
                vidx = len(self.virtual_nodes)
                self.virtual_nodes.append({"city": city, "gold": float(total_gold)})
                self.city_to_v.setdefault(city, []).append(vidx)

        self.cities: List[int] = sorted(self.city_to_v.keys())
        self.num_cities = len(self.cities)
        self.num_genes = len(self.virtual_nodes)

        self.real_ids = np.array([int(v["city"]) for v in self.virtual_nodes], dtype=int)
        self.golds = np.array([float(v["gold"]) for v in self.virtual_nodes], dtype=float)

        # precompute all-pairs shortest distances 

        adj = nx.to_scipy_sparse_array(self.graph, weight="dist", format="csr")
        self.dist_mat = scipy.sparse.csgraph.dijkstra(adj, directed=False)

        # best solution found
        self.best_city_order: Optional[List[int]] = None
        self.best_cost: float = float("inf")

        # cache for DP evaluation
        self._cost_cache: Dict[Tuple[int, ...], float] = {}

    # helper functions
    def _log(self, msg: str):
        if self.verbose:
            print(f"[LS] {msg}", flush=True)

    def _dist(self, u: int, v: int) -> float:
        return float(self.dist_mat[int(u), int(v)])

    def _choose_lookahead(self, n_genes: int) -> int:
        if self.beta <= 1.0:
            return min(n_genes, self.dp_lookahead_cap)
        return min(n_genes, 60)

    def _expand_city_order_to_perm(self, city_order: List[int]) -> List[int]:
        # keep all chunks of a city together (good for speed and usually good for cost)
        perm: List[int] = []
        for c in city_order:
            perm.extend(self.city_to_v[c])
        return perm

    # Split-DP evaluation on a permutation of virtual nodes
    def _evaluate_perm_cost(self, perm: List[int]) -> float:
        n = len(perm)
        V = [float("inf")] * (n + 1)
        V[0] = 0.0

        lookahead = self._choose_lookahead(n)

        for j in range(n):
            if not np.isfinite(V[j]):
                continue

            cost = 0.0
            w = 0.0
            u_real = 0  # depot

            limit = min(n, j + lookahead)
            for i in range(j, limit):
                idx = perm[i]
                v_real = int(self.real_ids[idx])
                d = self._dist(u_real, v_real)
                cost += d + (self.alpha * d * w) ** self.beta
                w += float(self.golds[idx])
                u_real = v_real

                d_home = self._dist(u_real, 0)
                total = cost + d_home + (self.alpha * d_home * w) ** self.beta

                cand = V[j] + total
                if cand < V[i + 1]:
                    V[i + 1] = cand

        return float(V[n])

    def _evaluate_city_order_cost(self, city_order: List[int]) -> float:
        key = tuple(city_order)
        if key in self._cost_cache:
            return self._cost_cache[key]

        perm = self._expand_city_order_to_perm(city_order)
        c = self._evaluate_perm_cost(perm)

        # simple cache cap to avoid memory blow-up
        if len(self._cost_cache) > 20000:
            self._cost_cache.clear()
        self._cost_cache[key] = float(c)
        return float(c)


    # Initialization: nearest neighbor over cities

    def _nn_city_order(self, random_k: int = 3) -> List[int]:
        unvisited = set(self.cities)
        cur = 0
        order: List[int] = []

        while unvisited:
            cand = list(unvisited)
            cand.sort(key=lambda x: self._dist(cur, x))
            if random_k > 1 and len(cand) > 1:
                k = min(random_k, len(cand))
                nxt = cand[self.rng.randrange(k)]
            else:
                nxt = cand[0]
            order.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        return order

    # Local search on CITY order (swap / 2-opt), evaluated by Split-DP
    def _neighbor(self, order: List[int]) -> List[int]:
        n = len(order)
        if n < 2:
            return order[:]

        # 70%: 2-opt, 30%: swap
        if self.rng.random() < 0.7:
            i = self.rng.randrange(0, n - 1)
            j = self.rng.randrange(i + 1, n)
            new = order[:]
            new[i:j] = reversed(new[i:j])
            return new
        else:
            i = self.rng.randrange(n)
            j = self.rng.randrange(n)
            while j == i:
                j = self.rng.randrange(n)
            new = order[:]
            new[i], new[j] = new[j], new[i]
            return new

    def optimize(self) -> Tuple[List[int], float]:
        start_time = time.time()

        # adaptive defaults for large instances
        if self.num_cities >= 900:
            restarts = min(self.max_iterations, 2)
            local_iters = min(self.max_local_iterations, 250)
        elif self.num_cities >= 500:
            restarts = min(self.max_iterations, 3)
            local_iters = min(self.max_local_iterations, 350)
        else:
            restarts = self.max_iterations
            local_iters = self.max_local_iterations

        best_order = None
        best_cost = float("inf")

        for r in range(restarts):
            if time.time() - start_time > self.time_limit_s:
                break

            # init
            order = self._nn_city_order(random_k=3 if r > 0 else 1)
            cur_cost = self._evaluate_city_order_cost(order)

            if cur_cost < best_cost:
                best_cost = cur_cost
                best_order = order[:]

            # hill-climb with random neighbors
            for _ in range(local_iters):
                if time.time() - start_time > self.time_limit_s:
                    break

                cand = self._neighbor(order)
                cand_cost = self._evaluate_city_order_cost(cand)

                if cand_cost + 1e-9 < cur_cost:
                    order = cand
                    cur_cost = cand_cost
                    if cur_cost + 1e-9 < best_cost:
                        best_cost = cur_cost
                        best_order = order[:]

        if best_order is None:
            best_order = self._nn_city_order(random_k=1)
            best_cost = self._evaluate_city_order_cost(best_order)

        self.best_city_order = best_order[:]
        self.best_cost = float(best_cost)
        return best_order, float(best_cost)

    # Benchmark 
    def solve(self, **kwargs):
        """
        Compatibility with your benchmark:
        - if 'generations' provided -> treat as restarts
        - you can also override time_limit_s/max_local_iterations/dp_lookahead_cap
        """
        if "generations" in kwargs:
            self.max_iterations = int(kwargs["generations"])
        if "max_iterations" in kwargs:
            self.max_iterations = int(kwargs["max_iterations"])
        if "max_local_iterations" in kwargs:
            self.max_local_iterations = int(kwargs["max_local_iterations"])
        if "time_limit_s" in kwargs:
            self.time_limit_s = float(kwargs["time_limit_s"])
        if "dp_lookahead_cap" in kwargs:
            self.dp_lookahead_cap = int(kwargs["dp_lookahead_cap"])
        if "verbose" in kwargs:
            self.verbose = bool(kwargs["verbose"])

        best_order, best_cost = self.optimize()
        # return (solution, cost) so your runner can unpack
        return best_order, float(best_cost)

    # Decode: Split-DP with predecessor, then output action list [(node, picked_gold), ...]
    def construct_full_path(self, city_order_or_tuple):
        # benchmark may pass solution or (solution, cost)
        if isinstance(city_order_or_tuple, tuple):
            city_order = city_order_or_tuple[0]
        else:
            city_order = city_order_or_tuple

        city_order = list(city_order)
        perm = self._expand_city_order_to_perm(city_order)

        n = len(perm)
        V = [float("inf")] * (n + 1)
        P = [0] * (n + 1)
        V[0] = 0.0

        lookahead = self._choose_lookahead(n)

        for j in range(n):
            if not np.isfinite(V[j]):
                continue

            cost = 0.0
            w = 0.0
            u_real = 0
            limit = min(n, j + lookahead)

            for i in range(j, limit):
                idx = perm[i]
                v_real = int(self.real_ids[idx])
                d = self._dist(u_real, v_real)
                cost += d + (self.alpha * d * w) ** self.beta
                w += float(self.golds[idx])
                u_real = v_real

                d_home = self._dist(u_real, 0)
                total = cost + d_home + (self.alpha * d_home * w) ** self.beta

                cand = V[j] + total
                if cand < V[i + 1]:
                    V[i + 1] = cand
                    P[i + 1] = j

        # recover trips as segments of perm
        trips: List[List[int]] = []
        cur = n
        while cur > 0:
            start = P[cur]
            trips.append(perm[start:cur])
            cur = start
        trips.reverse()

        actions: List[Tuple[int, float]] = []
        curr_real = 0

        for trip in trips:
            for vidx in trip:
                target = int(self.real_ids[vidx])
                gold = float(self.golds[vidx])

                if curr_real == target:
                    # multiple chunks at same city
                    actions.append((target, gold))
                else:
                    path = nx.shortest_path(self.graph, curr_real, target, weight="dist")
                    for node in path[1:]:
                        picked = gold if int(node) == target else 0.0
                        actions.append((int(node), float(picked)))
                    curr_real = target

            # return to depot
            if curr_real != 0:
                back = nx.shortest_path(self.graph, curr_real, 0, weight="dist")
                for node in back[1:]:
                    actions.append((int(node), 0.0))
                curr_real = 0

        # ensure end at depot
        if not actions or actions[-1] != (0, 0.0):
            actions.append((0, 0.0))
        return actions
