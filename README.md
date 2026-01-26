# Path Planning & Gold Collection Optimization

This project implements and compares various heuristic algorithms (SA, GA, SplitGA, SplitACO, LS) to solve the "City Visit Sequence" problem. The goal is to optimize the path for collecting gold and unloading at the depot (Node 0), balancing travel costs and load penalties.

## 🧠 Algorithms Overview

### 1. Simulated Annealing (SA)
* **Search Space:** Permutation space of "city visit sequences."
* **Mechanism:**
    * **Perturbation:** Generates neighbors using `swap` or `2-opt`.
    * **Dynamic Programming (Split-DP):** Automatically determines optimal split points for return trips to Node 0. It enumerates consecutive cities and selects partitions that minimize total cost.
* **Cost & Acceptance:**
    * Aligns with `Problem.cost`.
    * **Metropolis Criterion:** Unconditional acceptance of better solutions; worse solutions accepted with probability $e^{-\Delta/T}$.
    * **Cooling:** Geometric cooling 
* **Decoding:** Solutions are decoded into `[(node, gold), ..., (0, 0)]` using the shortest path. Gold is only collected at target cities.

### 2. Genetic Algorithm (GA)
* **Search Space:** Spatial search in city arrangement.
* **Fitness Evaluation:** Uses **Split-DP** with a `lookahead=20` to accelerate cost calculation.
* **Optimization:**
    * **Precomputation:** Uses `nx.all_pairs_dijkstra_path_length` to build a 2D `dist_matrix[u][v]` and `gold_list[node]` to eliminate dictionary lookup overhead.
    * **Operators:** Tournament selection, OX crossover, 2-opt/swap mutation, and elite retention.
    * **Initialization:** Smart initialization using angle-sweep injection.

### 3. SplitGA (🌟 Best Performer)
An enhanced version of GA specifically designed to handle high-penalty scenarios ($\beta > 1$).

* **Virtual Nodes:** When $\beta > 1$, large gold cities are split into multiple "virtual nodes" to manage load penalties effectively.
* **Gene Slicing:** Uses an automatic algorithm to slice gene sequences into multiple trips originating from/returning to 0.
* **Efficiency:** DP considers only the most recent 20 genes to reduce complexity.
* **Operators:** Nearest-neighbor initialization, tournament selection, OX crossover, interval reversal mutation, elite retention, and fitness saving.
* **Post-Processing:** Decodes trips into shortest-path action sequences with a greedy merge process to reduce unnecessary depot visits.
* **Theory:** With $\beta > 1$, the cost function becomes convex (higher weights = greater penalties). Splitting cities effectively linearizes the penalty.

### 4. SplitACO (Ant Colony Optimization)
* **Approach:** Shifted from node-level search to **city/virtual node-level search** to improve convergence.
* **Mechanism:** Avoids considering intermediate paths/cities that do not contain gold. Uses the destination as the guiding principle.
* **Decoding:** Finds the optimal node sequence first, then expands into the shortest execution path.

### 5. Local Search (LS)
* Utilizes the virtual node approach.
* Focuses on finding shortest paths within a local neighborhood to refine solutions.

---

## 📊 Summary & Analysis

**Usage:**
Run `s328249.py` to output the path and collected gold.
Because it consistently achieved the best results in our experiments, I selected SplitGA as the final solver to tackle the problem.
If you want to test additional methods and obtain cost/baseline comparisons, run `src/test.py`.

### Performance Insights
1.  **Linear Penalty ($\beta=1$):**
    * Improvement over baseline is minimal (0.0%–0.1%).
    * The baseline is already near-optimal for linear load penalties.
2.  **Quadratic Penalty ($\beta=2$):**
    * **SplitGA** dominates, achieving **70%–75% improvement**.
    * Success is attributed to chunking large gold cities into virtual nodes and using Split-DP to optimize return-to-base timing.
    * It delivers the best performance across all successfully executed cases.

### Engineering Considerations
* **Computational Cost:** SplitGA has a higher computational cost at $N=1000$ (approx. 570s).
* **Robustness:** `LS` and `MyACO` occasionally encounter `NetworkXNoPath` exceptions during decoding in $\beta=2$ cases. Future work requires enhanced robustness in path reconstruction.
* **Scalability:** `MyACO` suffers from excessive local search times in large-scale instances.

---

## 📈 Experimental Results

| Case | N | Density | $\alpha$ | $\beta$ | Baseline Cost | Best Method | Best Cost | Improvement | Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 100 | 0.2 | 1 | 1 | 25,266.4 | **SplitGA** | 25,235.0 | +0.1% | 0.59 |
| **2** | 100 | 0.2 | 2 | 1 | 50,425.3 | **SplitGA** | 50,393.1 | +0.1% | 0.65 |
| **3** | 100 | 0.2 | 1 | 2 | 5,334,401.9 | **SplitGA** | **1,560,420.8** | **+70.7%** | 4.23 |
| **4** | 100 | 1.0 | 1 | 1 | 18,266.2 | **SplitGA** | 18,253.6 | +0.1% | 1.18 |
| **5** | 100 | 1.0 | 2 | 1 | 36,457.9 | **SplitGA** | 36,448.2 | +0.0% | 1.09 |
| **6** | 100 | 1.0 | 1 | 2 | 5,404,978.1 | **SplitGA** | **1,333,976.4** | **+75.3%** | 7.81 |
| **7** | 1000 | 0.2 | 1 | 1 | 195,403.0 | **SplitGA** | 195,170.9 | +0.1% | 566.49 |
| **8** | 1000 | 0.2 | 2 | 1 | 390,028.7 | **SplitGA** | 389,816.3 | +0.1% | 573.52 |