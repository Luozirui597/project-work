import math
import random
import networkx as nx
import numpy as np
from tqdm import tqdm

class SplitGASolver:
    def __init__(self, problem):
        self.problem = problem
        self.graph = problem.graph
        self.alpha = problem.alpha
        self.beta = problem.beta
        
        # Data Preprocessing 
        real_cities = [n for n in self.graph.nodes if n != 0]
        
        # Virtual Node Splitting Strategy
        # Enable splitting when Beta > 1; when Beta = 1, splitting is generally not cost-effective
        # (and slows down execution), so it is set to disable.
        self.chunk_size = 200 if self.beta > 1.0 else float('inf')
        
        self.virtual_nodes = []
        for city_id in real_cities:
            total_gold = self.graph.nodes[city_id]['gold']
            if total_gold > self.chunk_size:
                rem = total_gold
                while rem > 0:
                    amt = min(rem, self.chunk_size)
                    self.virtual_nodes.append({'city_id': city_id, 'gold': amt})
                    rem -= amt
            else:
                self.virtual_nodes.append({'city_id': city_id, 'gold': total_gold})
                
        self.num_genes = len(self.virtual_nodes)
        
        # Pre-compute Distance Matrix
        # Use networkx to calculate all-pair distances and store them in a matrix to accelerate queries.
        raw_dist = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='dist'))
        max_id = max(self.graph.nodes) if self.graph.nodes else 0
        self.dist_mat = [[0.0]*(max_id+1) for _ in range(max_id+1)]
        for u in raw_dist:
            for v in raw_dist[u]:
                self.dist_mat[u][v] = raw_dist[u][v]

    def _get_dist(self, u_virt_idx, v_virt_idx):
        """ Obtain real distance between virtual nodes. """
        real_u = 0 if u_virt_idx == -1 else self.virtual_nodes[u_virt_idx]['city_id']
        real_v = 0 if v_virt_idx == -1 else self.virtual_nodes[v_virt_idx]['city_id']
        return self.dist_mat[real_u][real_v]

    def _split_cost(self, chromosome):
        """ Split DP Fitness Function. """
        n = len(chromosome)
        V = [float('inf')] * (n + 1)
        V[0] = 0
        lookahead = 20
        
        for j in range(n):
            if V[j] == float('inf'): continue
            limit = min(n, j + lookahead)
            cost = 0; load = 0; u = -1
            
            for i in range(j, limit):
                v_idx = chromosome[i]
                d = self._get_dist(u, v_idx)
                if d > 0: cost += d + (self.alpha * d * load) ** self.beta
                load += self.virtual_nodes[v_idx]['gold']
                u = v_idx
                
                d_home = self._get_dist(u, -1)
                return_cost = d_home + (self.alpha * d_home * load) ** self.beta
                
                total = cost + return_cost
                if V[j] + total < V[i+1]:
                    V[i+1] = V[j] + total
        return V[n]

    # Added: Helper Function for Trip Cost Calculation
    def _calculate_trip_cost(self, trip_tuples):
        """ Calculate the actual cost of a single Trip [(city_id, gold)...]. """
        cost = 0
        curr_w = 0
        curr_u = 0
        
        for node, gold in trip_tuples:
            # Accelerate using the pre-computed dist_mat
            d = self.dist_mat[curr_u][node]
            if d > 0:
                cost += d + (self.alpha * d * curr_w) ** self.beta
            curr_w += gold
            curr_u = node
            
        # Return to depot
        d_home = self.dist_mat[curr_u][0]
        cost += d_home + (self.alpha * d_home * curr_w) ** self.beta
        return cost

    #  Added: Aggressive Merge Post-processing (Merge Local Search) 
    def _post_process_merge(self, path):
        # Precompute required gold per real city
        required = {int(n): float(self.graph.nodes[n]['gold']) for n in self.graph.nodes if int(n) != 0}

        #Parse path into trips of PICKUPS ONLY (gold>0), split by depot visits
        trips = []
        current_trip = []

        # Remove consecutive duplicates to avoid noisy (0,0) sequences
        clean_path = []
        for step in path:
            if not clean_path or step != clean_path[-1]:
                clean_path.append(step)

        for node, gold in clean_path:
            node = int(node)
            gold = float(gold)
            if node == 0:
                if current_trip:
                    trips.append(current_trip)
                    current_trip = []
                continue

            # Only keep real pickup actions
            if gold > 0:
                current_trip.append((node, gold))

        if current_trip:
            trips.append(current_trip)

        if not trips:
            return [(0, 0.0)]

        # Greedy pairwise merge loop (same as before, but on pickup-only trips)
        improved = True
        while improved:
            improved = False
            best_merge = None
            best_saving = 0.0

            n_trips = len(trips)
            for i in range(n_trips):
                for j in range(n_trips):
                    if i == j:
                        continue

                    t1 = trips[i]
                    t2 = trips[j]

                    # Prune: only attempt small merges
                    if len(t1) + len(t2) > 20:
                        continue

                    cost_pre = self._calculate_trip_cost(t1) + self._calculate_trip_cost(t2)

                    merge_a = t1 + t2
                    cost_a = self._calculate_trip_cost(merge_a)

                    merge_b = t2 + t1
                    cost_b = self._calculate_trip_cost(merge_b)

                    saving_a = cost_pre - cost_a
                    saving_b = cost_pre - cost_b

                    current_max = saving_a if saving_a >= saving_b else saving_b
                    if current_max > 1e-6 and current_max > best_saving:
                        best_saving = current_max
                        best_trip = merge_a if saving_a >= saving_b else merge_b
                        best_merge = (i, j, best_trip)

            if best_merge:
                i, j, new_trip = best_merge
                # Remove old trips (delete higher index first to maintain order)
                idx_remove = sorted([i, j], reverse=True)
                trips.pop(idx_remove[0])
                trips.pop(idx_remove[1])
                trips.append(new_trip)
                improved = True

        # Reconstruct full traversal path from pickup trips, with gold clamping
        remaining = dict(required)  # Real city -> remaining gold to pick
        final_path = []

        for t in trips:
            curr_u = 0
            for target, planned_gold in t:
                target = int(target)
                planned_gold = float(planned_gold)

                # Travel along the shortest path
                seg = nx.shortest_path(self.graph, curr_u, target, weight='dist')
                for node in seg[1:]:
                    node = int(node)
                    g = 0.0
                    if node == target:
                        # Clamp by remaining requirement to avoid over-picking
                        rem = float(remaining.get(target, 0.0))
                        g = planned_gold if planned_gold <= rem else rem
                        remaining[target] = rem - g
                    final_path.append((node, g))
                curr_u = target

            # Return to depot
            seg_home = nx.shortest_path(self.graph, curr_u, 0, weight='dist')
            for node in seg_home[1:]:
                final_path.append((int(node), 0.0))

        return final_path

    def construct_full_path(self, chromosome):
        
        n = len(chromosome)
        V = [float('inf')] * (n + 1)
        P = [0] * (n + 1)
        V[0] = 0
        lookahead = 20
        
        # Split DP Decoding
        for j in range(n):
            if V[j] == float('inf'): continue
            cost = 0; load = 0; u = -1
            limit = min(n, j + lookahead)
            for i in range(j, limit):
                v_idx = chromosome[i]
                d = self._get_dist(u, v_idx)
                if d > 0: cost += d + (self.alpha * d * load) ** self.beta
                load += self.virtual_nodes[v_idx]['gold']
                u = v_idx
                d_home = self._get_dist(u, -1)
                return_cost = d_home + (self.alpha * d_home * load) ** self.beta
                if V[j] + cost + return_cost < V[i+1]:
                    V[i+1] = V[j] + cost + return_cost
                    P[i+1] = j

        trips = []
        curr = n
        while curr > 0:
            start = P[curr]
            trips.append(chromosome[start:curr])
            curr = start
        trips.reverse()
        
        #Generate Preliminary Path
        formatted_result = []
        curr_real_node = 0
        for trip in trips:
            for virt_idx in trip:
                target_real_node = self.virtual_nodes[virt_idx]['city_id']
                target_gold = self.virtual_nodes[virt_idx]['gold']
                path = nx.shortest_path(self.graph, curr_real_node, target_real_node, weight='dist')
                for node in path[1:]:
                    picked = target_gold if node == target_real_node else 0
                    formatted_result.append((node, picked))
                curr_real_node = target_real_node
            path_home = nx.shortest_path(self.graph, curr_real_node, 0, weight='dist')
            for node in path_home[1:]:
                formatted_result.append((node, 0))
            curr_real_node = 0
            
        # 3. Apply Aggressive Merge (Post-Process Merge)
        # This step attempts to merge Baseline-style one-way trips into more efficient paths.
        final_result = self._post_process_merge(formatted_result)
        
        return final_result

    # Smart Initialization Strategy (Kept Unchanged)
    def nearest_neighbor_init(self):
        unvisited = set(range(self.num_genes))
        chromosome = []
        curr_u = -1 
        while unvisited:
            best_v = None
            min_dist = float('inf')
            candidates = unvisited if len(unvisited) < 500 else random.sample(list(unvisited), 100)
            for v in candidates:
                d = self._get_dist(curr_u, v)
                if d < min_dist:
                    min_dist = d
                    best_v = v
            if best_v is None: best_v = unvisited.pop()
            else: unvisited.remove(best_v)
            chromosome.append(best_v)
            curr_u = best_v
        return chromosome

    def create_ind(self):
        ind = list(range(self.num_genes))
        random.shuffle(ind)
        return ind

    def crossover(self, p1, p2):
        size = len(p1)
        if size < 2: return p1
        a, b = sorted(random.sample(range(size), 2))
        child = [-1] * size
        child[a:b+1] = p1[a:b+1]
        ptr = 0
        present = set(p1[a:b+1])
        for gene in p2:
            if gene not in present:
                while ptr < size and child[ptr] != -1: ptr += 1
                if ptr < size: child[ptr] = gene
        return child

    def mutate(self, ind, rate=0.1):
        if random.random() < rate and len(ind) > 1:
            i, j = sorted(random.sample(range(len(ind)), 2))
            ind[i:j+1] = ind[i:j+1][::-1]
        return ind

    def solve(self, pop_size=50, generations=100, mutation_rate=0.2):
        population = []
        population.append(self.nearest_neighbor_init())
        while len(population) < pop_size:
            population.append(self.create_ind())
        
        fit_cache = {}
        def get_fit(ind):
            k = tuple(ind)
            if k not in fit_cache: fit_cache[k] = self._split_cost(ind)
            return fit_cache[k]

        for gen in tqdm(range(generations), desc="SplitGA Running", leave=False):
            fits = [get_fit(ind) for ind in population]
            new_pop = []
            
            sorted_idx = np.argsort(fits)
            new_pop.extend([population[i][:] for i in sorted_idx[:4]])
            
            while len(new_pop) < pop_size:
                candidates = random.sample(range(pop_size), 5)
                best_c = min(candidates, key=lambda i: fits[i])
                p1 = population[best_c]
                
                candidates = random.sample(range(pop_size), 5)
                best_c = min(candidates, key=lambda i: fits[i])
                p2 = population[best_c]
                
                child = self.crossover(p1, p2)
                child = self.mutate(child, mutation_rate)
                new_pop.append(child)
            
            population = new_pop

        final_fits = [get_fit(ind) for ind in population]
        return population[np.argmin(final_fits)]