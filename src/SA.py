import math
import random
import networkx as nx

class SimulatedAnnealingSolver:
    def __init__(self, problem):
        self.problem = problem
        self.graph = problem.graph
        self.alpha = problem.alpha
        self.beta = problem.beta
        
        # Retrieve all city nodes containing gold (excluding the depot at Node 0)
        self.cities = [n for n in self.graph.nodes if n != 0]
        
        # Pre-compute the shortest path distances between all pairs of nodes
        self.dist_matrix = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='dist'))

    def _calculate_segment_cost(self, i, j, solution):
        """ Calculate the cost of a sub-path (logic remains unchanged). """
        cost = 0
        current_weight = 0
        prev_node = 0
        
        for k in range(i, j + 1):
            curr_node = solution[k]
            d = self.dist_matrix[prev_node][curr_node]
            if d > 0:
                cost += d + (self.alpha * d * current_weight) ** self.beta
            current_weight += self.graph.nodes[curr_node]['gold']
            prev_node = curr_node
            
        d_home = self.dist_matrix[prev_node][0]
        cost += d_home + (self.alpha * d_home * current_weight) ** self.beta
        return cost

    def evaluate_split(self, solution):
        """ The Split algorithm (logic remains unchanged). """
        n = len(solution)
        V = [float('inf')] * (n + 1)
        V[0] = 0
        
        for i in range(1, n + 1):
            lookback_limit = 20 
            start_j = max(0, i - lookback_limit)
            for j in range(start_j, i):
                trip_cost = self._calculate_segment_cost(j, i-1, solution)
                if V[j] + trip_cost < V[i]:
                    V[i] = V[j] + trip_cost
        return V[n]

    def construct_full_path(self, solution):
        """ Reconstruct the full traversal path based on the optimal splits. """
        n = len(solution)
        V = [float('inf')] * (n + 1)
        P = [0] * (n + 1)
        V[0] = 0
        
        # Re-run the Dynamic Programming (DP) process to retrieve the split points
        for i in range(1, n + 1):
            lookback_limit = 20
            start_j = max(0, i - lookback_limit)
            for j in range(start_j, i):
                trip_cost = self._calculate_segment_cost(j, i-1, solution)
                if V[j] + trip_cost < V[i]:
                    V[i] = V[j] + trip_cost
                    P[i] = j 
        
        # Extract path segments (Trips)
        trips = []
        curr = n
        while curr > 0:
            start_idx = P[curr]
            segment = solution[start_idx:curr]
            trips.append(segment)
            curr = start_idx
        trips.reverse()
        
        # Construct the detailed traversal path
        formatted_result = [] # Stores (node, gold) tuples
        current_node = 0 
        
        for trip in trips:
            # Each trip follows the sequence: 0 -> target1 -> target2 -> ... -> 0
            
            for target in trip:
                # A. Move to the next target node (Path: current -> ... -> target)
                path_segment = nx.shortest_path(self.graph, current_node, target, weight='dist')
                
                # Iterate through nodes on the path (skipping the first, as the starting point is already recorded)
                for node in path_segment[1:]:
                    if node == target:
                        # Arrived at the target node: collect the gold
                        gold = self.graph.nodes[node]['gold']
                        formatted_result.append((node, gold))
                    else:
                        # Passing through only: do not collect gold (gold=0) to minimize weight penalties
                        formatted_result.append((node, 0))
                
                current_node = target
            
            # B. End of this trip; return to the depot (Path: current -> ... -> 0)
            path_home = nx.shortest_path(self.graph, current_node, 0, weight='dist')
            for node in path_home[1:]:
                # All nodes on the return journey (including the depot at Node 0) are recorded as 0
                formatted_result.append((node, 0))
            
            current_node = 0
            
        return formatted_result

    def get_neighbor(self, solution):
        """ Neighborhood operations (perturbation). """
        new_solution = solution[:]
        if len(solution) < 2: return new_solution
        idx1, idx2 = random.sample(range(len(solution)), 2)
        if random.random() < 0.5:
            new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]
        else:
            start, end = min(idx1, idx2), max(idx1, idx2)
            new_solution[start:end+1] = reversed(new_solution[start:end+1])
        return new_solution

    def solve(self, initial_temp=2000, cooling_rate=0.995, max_iter=3000):
        """ Main execution loop. """
        current_solution = list(self.cities)
        random.shuffle(current_solution)
        current_cost = self.evaluate_split(current_solution)
        best_solution = current_solution[:]
        best_cost = current_cost
        temp = initial_temp
        
        for i in range(max_iter):
            neighbor = self.get_neighbor(current_solution)
            neighbor_cost = self.evaluate_split(neighbor)
            delta = neighbor_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_solution = neighbor
                current_cost = neighbor_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = current_solution[:]
            temp *= cooling_rate
            if temp < 1e-8: break
            
        return best_solution