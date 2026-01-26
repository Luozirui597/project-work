import time
import random
from typing import List, Tuple, Optional

import numpy as np
import networkx as nx
import scipy.sparse
import scipy.sparse.csgraph
from tqdm.auto import tqdm


class RankBasedACOSolver:

    def __init__(self, problem):
        self.problem = problem
        # problem.graph returns a copy; safe to keep locally
        self.graph: nx.Graph = problem.graph
        self.phys_alpha = float(problem.alpha)
        self.phys_beta = float(problem.beta)

        # Virtual node strategy
        # Chunk only when beta > 1 
        real_cities = [int(n) for n in self.graph.nodes if int(n) != 0]

        if self.phys_beta > 1.0:
            self.chunk_size = 500.0
            nodes = []
            for city_id in real_cities:
                total_gold = float(self.graph.nodes[city_id]["gold"])
                if total_gold > self.chunk_size:
                    rem = total_gold
                    while rem > 0:
                        amt = min(rem, self.chunk_size)
                        nodes.append({"id": int(city_id), "gold": float(amt)})
                        rem -= amt
                else:
                    nodes.append({"id": int(city_id), "gold": float(total_gold)})
            self.nodes = nodes
        else:
            self.chunk_size = None
            self.nodes = [{"id": int(c), "gold": float(self.graph.nodes[c]["gold"])} for c in real_cities]

        self.num_genes = len(self.nodes)
        self.real_ids = np.array([n["id"] for n in self.nodes], dtype=int)
        self.golds = np.array([n["gold"] for n in self.nodes], dtype=float)

        
        # 2 Precompute all-pairs shortest distances + predecessors
        n_nodes = self.graph.number_of_nodes()
        do_print = (n_nodes >= 250)
        t0 = time.time()
        if do_print:
            print(f"[MyACO] Precomputing all-pairs shortest distances for |V|={n_nodes} ...", flush=True)

        # Problem nodes are 0..N-1 contiguous; adjacency index = node id
        adj = nx.to_scipy_sparse_array(self.graph, weight="dist", format="csr")
        self.dist_mat, self.pred = scipy.sparse.csgraph.dijkstra(
            adj, directed=False, return_predecessors=True
        )

        if do_print:
            print(f"[MyACO] Distances ready. Time: {time.time() - t0:.2f}s", flush=True)

        # Heuristic + pheromones
        sub_dist = self.dist_mat[self.real_ids][:, self.real_ids].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            heuristic = 1.0 / (sub_dist + 1e-9)
        np.fill_diagonal(heuristic, 0.0)

        # If chunking creates same-city duplicates, sub_dist may be 0 off-diagonal.
        # Replace those with a typical distance scale.
        same_city = (self.real_ids[:, None] == self.real_ids[None, :])
        offdiag = ~np.eye(self.num_genes, dtype=bool)
        same_city_offdiag = same_city & offdiag
        positive = sub_dist[(sub_dist > 0) & np.isfinite(sub_dist)]
        typical = float(np.median(positive)) if positive.size > 0 else 1.0
        heuristic[same_city_offdiag] = 1.0 / (typical + 1e-9)

        self.heuristic_mat = heuristic
        self.pheromones = np.ones((self.num_genes, self.num_genes), dtype=float) * 0.01

    # Dist helper (virtual -> real)
    def _get_dist(self, u_idx: int, v_idx: int) -> float:
        real_u = 0 if u_idx == -1 else int(self.real_ids[u_idx])
        real_v = 0 if v_idx == -1 else int(self.real_ids[v_idx])
        return float(self.dist_mat[real_u, real_v])

    # Path reconstruction using predecessor matrix (FAST, NO nx.shortest_path)
    def _reconstruct_path(self, u: int, v: int) -> Optional[List[int]]:
        if u == v:
            return [u]
        # unreachable if pred[v,u] is -9999; scipy uses -9999 for "no predecessor"
        # We reconstruct by walking backwards from v using pred[u, *]
        cur = int(v)
        path_rev = [cur]

        # Safety bound: at most n_nodes steps
        n_nodes = self.pred.shape[0]
        for _ in range(n_nodes):
            cur_pred = int(self.pred[int(u), cur])
            if cur_pred < 0:
                return None
            path_rev.append(cur_pred)
            if cur_pred == u:
                break
            cur = cur_pred
        else:
            return None

        path_rev.reverse()
        if path_rev[0] != u or path_rev[-1] != v:
            return None
        return path_rev

    # Trip cost for a single trip visiting given virtual indices and returning to 0
    def _calculate_trip_cost_indices(self, indices: List[int]) -> float:
        cost = 0.0
        w = 0.0
        u = -1
        for idx in indices:
            d = self._get_dist(u, idx)
            if not np.isfinite(d):
                return float("inf")
            cost += d + (self.phys_alpha * d * w) ** self.phys_beta
            w += float(self.golds[idx])
            u = idx
        d_home = self._get_dist(u, -1)
        if not np.isfinite(d_home):
            return float("inf")
        cost += d_home + (self.phys_alpha * d_home * w) ** self.phys_beta
        return float(cost)

    # Split-DP lookahead
    def _choose_lookahead(self, n: int, dp_lookahead_cap: int = 200) -> int:
        # For beta<=1, full O(n^2) inside ACO is expensive -> cap it
        if self.phys_beta <= 1.0:
            return min(n, int(dp_lookahead_cap))
        return min(n, 60)

    # Split-DP evaluation: split permutation into multiple depot-ended trips
    def _evaluate_cost(self, permutation: List[int], dp_lookahead_cap: int = 200) -> float:
        n = len(permutation)
        V = [float("inf")] * (n + 1)
        V[0] = 0.0

        lookahead = self._choose_lookahead(n, dp_lookahead_cap=dp_lookahead_cap)

        for j in range(n):
            if not np.isfinite(V[j]):
                continue

            cost = 0.0
            weight = 0.0
            u = -1
            limit = min(n, j + lookahead)

            for i in range(j, limit):
                v_idx = permutation[i]
                d = self._get_dist(u, v_idx)
                if not np.isfinite(d):
                    break  # cannot continue this segment
                cost += d + (self.phys_alpha * d * weight) ** self.phys_beta
                weight += float(self.golds[v_idx])
                u = v_idx

                d_home = self._get_dist(u, -1)
                if not np.isfinite(d_home):
                    continue  # cannot return to depot from here
                total = cost + d_home + (self.phys_alpha * d_home * weight) ** self.phys_beta

                cand = V[j] + total
                if cand < V[i + 1]:
                    V[i + 1] = cand

        return float(V[n])

    # Initialization: fast NN on virtual nodes (real distance)
    def _nn_init_perm(self) -> List[int]:
        n = self.num_genes
        remaining = np.ones(n, dtype=bool)
        perm: List[int] = []

        cur = random.randrange(n)
        perm.append(int(cur))
        remaining[cur] = False

        for _ in range(n - 1):
            unv = np.flatnonzero(remaining)
            cur_real = int(self.real_ids[cur])
            cand_real = self.real_ids[unv]
            dists = self.dist_mat[cur_real, cand_real].astype(float)

            # Choose nearest reachable
            # graph is connected, but keep robust anyway
            order = np.argsort(dists)
            nxt = None
            for k in order:
                if np.isfinite(dists[k]):
                    nxt = int(unv[int(k)])
                    break
            if nxt is None:
                # fallback: any remaining
                nxt = int(unv[0])

            perm.append(nxt)
            remaining[nxt] = False
            cur = nxt

        return perm

    # Simple local search on permutation using capped DP eval
    def _two_opt_perm(self, perm: List[int], max_rounds: int = 2, dp_lookahead_cap: int = 200):
        best = perm[:]
        best_cost = self._evaluate_cost(best, dp_lookahead_cap=dp_lookahead_cap)
        n = len(best)

        for _ in range(max_rounds):
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n):
                    if j - i == 1:
                        continue
                    cand = best[:i] + best[i:j][::-1] + best[j:]
                    c = self._evaluate_cost(cand, dp_lookahead_cap=dp_lookahead_cap)
                    if c + 1e-9 < best_cost:
                        best, best_cost = cand, c
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best, float(best_cost)

    def _relocate_perm(self, perm: List[int], max_rounds: int = 1, dp_lookahead_cap: int = 200):
        best = perm[:]
        best_cost = self._evaluate_cost(best, dp_lookahead_cap=dp_lookahead_cap)
        n = len(best)

        for _ in range(max_rounds):
            improved = False
            for i in range(n):
                node = best[i]
                rest = best[:i] + best[i + 1 :]
                for k in range(n):
                    if k == i:
                        continue
                    cand = rest[:k] + [node] + rest[k:]
                    c = self._evaluate_cost(cand, dp_lookahead_cap=dp_lookahead_cap)
                    if c + 1e-9 < best_cost:
                        best, best_cost = cand, c
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best, float(best_cost)

    # Decode into (node, picked_gold) action list using predecessor matrix
    def construct_full_path(self, permutation: List[int], dp_lookahead_cap: int = 200) -> List[Tuple[int, float]]:
        n = len(permutation)
        V = [float("inf")] * (n + 1)
        P = [0] * (n + 1)
        V[0] = 0.0

        lookahead = self._choose_lookahead(n, dp_lookahead_cap=dp_lookahead_cap)

        for j in range(n):
            if not np.isfinite(V[j]):
                continue

            cost = 0.0
            weight = 0.0
            u = -1
            limit = min(n, j + lookahead)

            for i in range(j, limit):
                v_idx = permutation[i]
                d = self._get_dist(u, v_idx)
                if not np.isfinite(d):
                    break
                cost += d + (self.phys_alpha * d * weight) ** self.phys_beta
                weight += float(self.golds[v_idx])
                u = v_idx

                d_home = self._get_dist(u, -1)
                if not np.isfinite(d_home):
                    continue
                total = cost + d_home + (self.phys_alpha * d_home * weight) ** self.phys_beta

                cand = V[j] + total
                if cand < V[i + 1]:
                    V[i + 1] = cand
                    P[i + 1] = j

        # Recover trip segments (virtual indices)
        trips: List[List[int]] = []
        cur = n
        while cur > 0:
            start = P[cur]
            trips.append(permutation[start:cur])
            cur = start
        trips.reverse()

        actions: List[Tuple[int, float]] = []
        curr_real = 0

        for trip in trips:
            for idx in trip:
                target = int(self.real_ids[idx])
                gold = float(self.golds[idx])

                # reconstruct shortest path curr_real -> target
                if curr_real == target:
                    actions.append((target, gold))
                else:
                    path = self._reconstruct_path(curr_real, target)
                    if path is None:
                        # should not happen since Problem is connected, but keep safe
                        return [(0, 0.0)]
                    for node in path[1:]:
                        picked = gold if int(node) == target else 0.0
                        actions.append((int(node), float(picked)))
                    curr_real = target

            # return to depot (unload)
            if curr_real != 0:
                path0 = self._reconstruct_path(curr_real, 0)
                if path0 is None:
                    return [(0, 0.0)]
                for node in path0[1:]:
                    actions.append((int(node), 0.0))
                curr_real = 0

        if not actions or actions[-1] != (0, 0.0):
            actions.append((0, 0.0))
        return actions

    # Main ACO solve
    def solve(
        self,
        n_ants: int = 20,
        generations: int = 50,
        rho: float = 0.1,
        aco_alpha: float = 1.0,
        aco_beta: float = 2.0,
        elitism: int = 5,
        explore_eps: float = 0.02,
        dp_lookahead_cap: int = 200,
        verbose: bool = True,
        do_local_search: bool = True,
    ) -> List[int]:
        fit_cache = {}

        def get_cost(sol: List[int]) -> float:
            t = tuple(sol)
            if t not in fit_cache:
                fit_cache[t] = self._evaluate_cost(sol, dp_lookahead_cap=dp_lookahead_cap)
            return float(fit_cache[t])

        t0 = time.time()
        if verbose:
            print(
                f"[MyACO] solve() start: genes={self.num_genes}, ants={n_ants}, gens={generations}, beta={self.phys_beta}",
                flush=True,
            )

        #init
        best_perm = self._nn_init_perm()
        best_cost = self._evaluate_cost(best_perm, dp_lookahead_cap=dp_lookahead_cap)

        if verbose:
            print(f"[MyACO] init done. best_cost={best_cost:.3f} (time {time.time()-t0:.2f}s)", flush=True)

        # seed pheromone
        seed_deposit = 10.0
        for i in range(len(best_perm) - 1):
            u, v = best_perm[i], best_perm[i + 1]
            self.pheromones[u, v] += seed_deposit
            self.pheromones[v, u] += seed_deposit

        dynamic_Q = None
        all_idx = np.arange(self.num_genes, dtype=int)

        #ACO loop
        if verbose:
            print("[MyACO] ACO loop start ...", flush=True)

        gen_iter = tqdm(range(generations), desc="FastRACO", leave=True, dynamic_ncols=True, disable=False)

        for _gen in gen_iter:
            solutions = []

            for _ in range(n_ants):
                current = random.randrange(self.num_genes)
                visited_mask = np.zeros(self.num_genes, dtype=bool)
                visited_mask[current] = True
                path = [int(current)]

                w_cur = 0.0

                while len(path) < self.num_genes:
                    unvisited = all_idx[~visited_mask]

                    tau = self.pheromones[current, unvisited].astype(float)

                    cur_real = int(self.real_ids[current])
                    unv_real = self.real_ids[unvisited]
                    dists = self.dist_mat[cur_real, unv_real].astype(float)
                    cand_gold = self.golds[unvisited].astype(float)

                    # beta<=1: prefer delaying heavy pickups slightly
                    if self.phys_beta <= 1.0:
                        gold_bias = 0.6
                        approx = dists * (1.0 + self.phys_alpha * (w_cur + gold_bias * cand_gold))
                    else:
                        approx = dists + (self.phys_alpha * dists * w_cur) ** self.phys_beta

                    eta = 1.0 / (approx + 1e-9)

                    probs = (tau ** float(aco_alpha)) * (eta ** float(aco_beta))
                    s = probs.sum()

                    if (not np.isfinite(s)) or s <= 0.0 or random.random() < explore_eps:
                        nxt = int(np.random.choice(unvisited))
                    else:
                        probs = probs / s
                        nxt = int(np.random.choice(unvisited, p=probs))

                    path.append(nxt)
                    visited_mask[nxt] = True
                    w_cur += float(self.golds[nxt])
                    current = nxt

                c = get_cost(path)
                solutions.append((path, c))

            solutions.sort(key=lambda x: x[1])
            gen_best_path, gen_best_cost = solutions[0]

            if gen_best_cost + 1e-9 < best_cost:
                best_cost = float(gen_best_cost)
                best_perm = gen_best_path[:]

            # evaporation
            self.pheromones *= (1.0 - float(rho))

            # dynamic Q
            if dynamic_Q is None:
                dynamic_Q = 0.5 * best_cost / max(int(elitism), 1)

            # rank-based deposit
            rank_limit = min(int(elitism), len(solutions))
            for rank in range(rank_limit):
                sol, sol_cost = solutions[rank]
                w_rank = (rank_limit - rank)
                dep = (float(dynamic_Q) / max(float(sol_cost), 1e-12)) * float(w_rank)
                for i in range(len(sol) - 1):
                    u, v = sol[i], sol[i + 1]
                    self.pheromones[u, v] += dep
                    self.pheromones[v, u] += dep

            gen_iter.set_postfix(best=f"{best_cost:.2f}")

        if verbose:
            print(f"[MyACO] ACO loop done. best_cost={best_cost:.3f} (time {time.time()-t0:.2f}s)", flush=True)

        #final local search
        if do_local_search:
            if verbose:
                print("[MyACO] local search start (2-opt / relocate) ...", flush=True)
            best_perm, best_cost = self._two_opt_perm(best_perm, max_rounds=2, dp_lookahead_cap=dp_lookahead_cap)
            best_perm, best_cost = self._relocate_perm(best_perm, max_rounds=1, dp_lookahead_cap=dp_lookahead_cap)
            if verbose:
                print(f"[MyACO] local search done. best_cost={best_cost:.3f} (time {time.time()-t0:.2f}s)", flush=True)
        else:
            if verbose:
                print("[MyACO] local search skipped.", flush=True)

        return best_perm
