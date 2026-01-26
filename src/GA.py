import math
import random
import networkx as nx
import numpy as np
from tqdm import tqdm

class GeneticAlgorithmSolver:
    def __init__(self, problem):
        self.problem = problem
        self.graph = problem.graph
        
        # Data preprocessing
        self.cities = [n for n in self.graph.nodes if n != 0]
        self.num_cities = len(self.cities)
        self.alpha = problem.alpha
        self.beta = problem.beta
        
        # Flatten data structures for acceleration
        max_node_id = max(self.graph.nodes) if self.graph.nodes else 0
        self.gold_list = [0.0] * (max_node_id + 1)
        gold_dict = nx.get_node_attributes(self.graph, 'gold')
        for n, g in gold_dict.items():
            self.gold_list[n] = g
            
        # print("Pre-computing all-pairs shortest paths (Dijkstra)...")
        # Use networkx's built-in efficient algorithm to obtain all distances
        raw_dist = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='dist'))
        self.dist_matrix = [[0.0] * (max_node_id + 1) for _ in range(max_node_id + 1)]
        for u in raw_dist:
            for v in raw_dist[u]:
                self.dist_matrix[u][v] = raw_dist[u][v]
        # print("Pre-computation complete.")

    def _split_cost(self, solution):
        """ 
        Split Algorithm: Calculate the cost (Fitness) of the current sequence under optimal splitting.
        """
        n = len(solution)
        V = [float('inf')] * (n + 1)
        V[0] = 0
        
        # Lookahead steps
        lookahead = 20 
        
        dist_mat = self.dist_matrix
        gold_lst = self.gold_list
        alpha = self.alpha
        beta = self.beta
        
        for j in range(n):
            if V[j] == float('inf'): continue
            current_load = 0
            trip_cost = 0
            u = 0 # Base / Depot
            
            limit = min(n, j + lookahead)
            for i in range(j, limit):
                v = solution[i]
                # Outbound journey (u->v)
                dist = dist_mat[u][v]
                if dist > 0:
                    trip_cost += dist + (alpha * dist * current_load) ** beta
                
                # Load gold
                current_load += gold_lst[v]
                u = v
                
                # Return journey (v->0)
                dist_home = dist_mat[u][0]
                return_cost = dist_home + (alpha * dist_home * current_load) ** beta
                
                # State transition
                total = trip_cost + return_cost
                if V[j] + total < V[i+1]:
                    V[i+1] = V[j] + total
        return V[n]

    def construct_full_path(self, solution):
        # reconstruct the full traversal path based on the optimal splits
        n = len(solution)
        V = [float('inf')] * (n + 1)
        P = [0] * (n + 1) # Record the predecessor of the split point
        V[0] = 0
        
        lookahead = 20
        dist_mat = self.dist_matrix
        gold_lst = self.gold_list
        alpha = self.alpha
        beta = self.beta
        
        # 1. Re-execute the DP logic to record path P
        for j in range(n):
            if V[j] == float('inf'): continue
            current_load = 0
            trip_cost = 0
            u = 0 
            
            limit = min(n, j + lookahead)
            for i in range(j, limit):
                v = solution[i]
                dist = dist_mat[u][v]
                if dist > 0:
                    trip_cost += dist + (alpha * dist * current_load) ** beta
                
                current_load += gold_lst[v]
                u = v
                
                dist_home = dist_mat[u][0]
                return_cost = dist_home + (alpha * dist_home * current_load) ** beta
                
                total = trip_cost + return_cost
                if V[j] + total < V[i+1]:
                    V[i+1] = V[j] + total
                    P[i+1] = j # Record that this segment started from j
        
        # 2. Backtrack to extract trips
        trips = []
        curr = n
        while curr > 0:
            start_idx = P[curr]
            trips.append(solution[start_idx:curr])
            curr = start_idx
        trips.reverse()
        
        # 3. Generate detailed path nodes
        formatted_result = []
        current_node = 0
        
        for trip in trips:
            # Trip: 0 -> target1 -> target2 -> ... -> 0
            for target in trip:
                # Find specific path nodes (need to use networkx to find intermediate points)
                path_segment = nx.shortest_path(self.graph, current_node, target, weight='dist')
                
                # Iterate through the path segment (skip the first point as it is already recorded)
                for node in path_segment[1:]:
                    if node == target:
                        # Target point: Collect gold
                        gold = self.graph.nodes[node]['gold']
                        formatted_result.append((node, gold))
                    else:
                        # Passing point: Do not collect gold
                        formatted_result.append((node, 0))
                current_node = target
            
            # Return journey
            path_home = nx.shortest_path(self.graph, current_node, 0, weight='dist')
            for node in path_home[1:]:
                formatted_result.append((node, 0))
            current_node = 0
            
        return formatted_result

    #GA Initialization
    def smart_initialize(self):
        """ Smart initialization: Nearest Neighbor based on angles from the depot. """
        base_pos = self.graph.nodes[0]['pos']
        city_angles = []
        for c in self.cities:
            pos = self.graph.nodes[c]['pos']
            angle = math.atan2(pos[1] - base_pos[1], pos[0] - base_pos[0])
            city_angles.append((angle, c))
        city_angles.sort()
        return [c for _, c in city_angles]

    def create_individual(self):
        ind = list(self.cities)
        random.shuffle(ind)
        return ind

    # Genetic Operators-
    def crossover_ox(self, parent1, parent2):
        size = len(parent1)
        if size < 2: return parent1
        
        try:
            a, b = sorted(random.sample(range(size), 2))
        except ValueError:
            return parent1
            
        child = [None] * size
        child[a:b+1] = parent1[a:b+1]
        
        ptr = 0
        current_gene_set = set(parent1[a:b+1])
        
        for gene in parent2:
            if gene not in current_gene_set:
                while ptr < size and child[ptr] is not None:
                    ptr += 1
                if ptr < size:
                    child[ptr] = gene
        return child

    def mutate(self, individual, rate=0.1):
        if random.random() < rate and len(individual) > 1:
            if random.random() < 0.7:
                i, j = sorted(random.sample(range(len(individual)), 2))
                individual[i:j+1] = individual[i:j+1][::-1]
            else:
                i, j = random.sample(range(len(individual)), 2)
                individual[i], individual[j] = individual[j], individual[i]
        return individual

    def tournament_selection(self, population, fitnesses, k=3):
        selected_indices = random.sample(range(len(population)), k)
        best_idx = selected_indices[0]
        for idx in selected_indices[1:]:
            if fitnesses[idx] < fitnesses[best_idx]:
                best_idx = idx
        return population[best_idx]

    # main
    def solve(self, pop_size=100, generations=200, mutation_rate=0.2, elitism_size=2):
        #Initialize population
        population = [self.create_individual() for _ in range(pop_size)]
        
        # Inject smart individual
        if population:
            smart_ind = self.smart_initialize()
            population[0] = smart_ind
        
        fitness_cache = {}
        def get_fitness(ind):
            key = tuple(ind)
            if key not in fitness_cache:
                fitness_cache[key] = self._split_cost(ind)
            return fitness_cache[key]

        fitnesses = [get_fitness(ind) for ind in population]
        
        best_idx = np.argmin(fitnesses)
        global_best_ind = list(population[best_idx])
        global_best_cost = fitnesses[best_idx]
        
        # print(f"Initial population best cost: {global_best_cost:.2f}")

        # Iteration (use tqdm to show progress)
        # If you do not need to see the progress bar, you can change tqdm(range(...)) to range(...)
        iterator = range(generations)
        # iterator = tqdm(range(generations), desc="GA Evolution", leave=False) 
        
        for gen in iterator:
            new_population = []
            
            # Elite retention
            sorted_indices = np.argsort(fitnesses)
            for i in range(elitism_size):
                new_population.append(list(population[sorted_indices[i]]))
            
            # Reproduction
            while len(new_population) < pop_size:
                p1 = self.tournament_selection(population, fitnesses)
                p2 = self.tournament_selection(population, fitnesses)
                
                child = self.crossover_ox(p1, p2)
                child = self.mutate(child, mutation_rate)
                new_population.append(child)
            
            population = new_population
            fitnesses = [get_fitness(ind) for ind in population]
            
            # Update global optimum
            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < global_best_cost:
                global_best_cost = fitnesses[gen_best_idx]
                global_best_ind = list(population[gen_best_idx])
                
        return global_best_ind 