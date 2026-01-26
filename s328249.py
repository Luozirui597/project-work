import os
import sys
import time
from typing import List, Tuple, Callable, Any, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Required by spec: import Problem from Problem.py
from Problem import Problem

# Prefer: src is a package
try:
    from src.SplitACO import RankBasedACOSolver
    from src.SplitGA import SplitGASolver
    from src.GA import GeneticAlgorithmSolver
    from src.SA import SimulatedAnnealingSolver
    from src.LS import LocalSearch
except ImportError as e:
    raise ImportError(
        "Import failed. Please ensure:\n"
        "  - Repository structure: project-work/Problem.py, project-work/s328249.py, project-work/src/\n"
        "  - src contains: SplitACO.py, SplitGA.py, GA.py, SA.py, LS.py\n"
        "  - src/__init__.py exists (empty file)\n"
        "And the class names are:\n"
        "  RankBasedACOSolver, SplitGASolver, GeneticAlgorithmSolver, SimulatedAnnealingSolver, LocalSearchOptimizer"
    ) from e


def compute_total_cost(problem: Problem, actions: List[Tuple[int, float]]) -> float:
    if not actions:
        return float("inf")

    total = 0.0
    carried = 0.0

    # The actions list represents visited nodes in order
    prev_node = actions[0][0]

    # If first action is not 0, we still handle it, but typical decoders start from 0.
    # Gold in the first element is picked at start node (rare). Apply it now.
    first_gold = float(actions[0][1])
    if prev_node == 0:
        carried = 0.0
    if first_gold > 0:
        carried += first_gold

    for (node, gold) in actions[1:]:
        node = int(node)
        gold = float(gold)

        # travel prev_node -> node with current carried weight
        total += problem.cost([prev_node, node], carried)

        # arrive: pick gold if any
        if gold > 0:
            carried += gold

        # unload if depot
        if node == 0:
            carried = 0.0

        prev_node = node

    return float(total)


def print_algo_result(name: str, actions: List[Tuple[int, float]], cost: float, elapsed_s: float):
    print("\n" + "=" * 80)
    print(f"[{name}]")
    print("-" * 80)
    print(f"Total cost: {cost:.6f}")
    print(f"Elapsed:    {elapsed_s:.4f} s")
    print(f"Steps:      {len(actions)}")
    if actions:
        print(f"Last step:  {actions[-1]}")
    print("-" * 80)
    print("FULL PATH BEGIN")
    print(actions)
    print("FULL PATH END")
    print("=" * 80)



def run_solver(
    name: str,
    build_solver: Callable[[Problem], Any],
    solve_fn: Callable[[Any], Any],
    decode_fn: Callable[[Any, Any], List[Tuple[int, float]]],
    problem: Problem,
) -> Tuple[List[Tuple[int, float]], float, float]:
    solver = build_solver(problem)
    t0 = time.time()
    sol = solve_fn(solver)
    actions = decode_fn(solver, sol)
    elapsed = time.time() - t0
    cost = compute_total_cost(problem, actions)
    return actions, cost, elapsed


def solution(p: Problem):
    solver = RankBasedACOSolver(p)
    best_perm = solver.solve(
        n_ants=20,
        generations=50,
        do_local_search=True,
        verbose=False,
    )
    actions = solver.construct_full_path(best_perm)
    return actions


if __name__ == "__main__":
    # Create a test problem (example parameters)
    test_problem = Problem(num_cities=20, alpha=1.0, beta=1.0, density=0.5)

    # A) SplitACO
    actions, cost, elapsed = run_solver(
        name="SplitACO (RankBasedACOSolver)",
        build_solver=lambda p: RankBasedACOSolver(p),
        solve_fn=lambda s: s.solve(n_ants=20, generations=50, do_local_search=True, verbose=False),
        decode_fn=lambda s, sol: s.construct_full_path(sol),
        problem=test_problem,
    )
    print_algo_result("SplitACO (RankBasedACOSolver)", actions, cost, elapsed)

    # B) SplitGA
    actions, cost, elapsed = run_solver(
        name="SplitGA (SplitGASolver)",
        build_solver=lambda p: SplitGASolver(p),
        solve_fn=lambda s: s.solve(pop_size=50, generations=100, mutation_rate=0.2),
        decode_fn=lambda s, sol: s.construct_full_path(sol),
        problem=test_problem,
    )
    print_algo_result("SplitGA (SplitGASolver)", actions, cost, elapsed)

    # C) GA
    actions, cost, elapsed = run_solver(
        name="GA (GeneticAlgorithmSolver)",
        build_solver=lambda p: GeneticAlgorithmSolver(p),
        solve_fn=lambda s: s.solve(pop_size=100, generations=200, mutation_rate=0.2, elitism_size=2),
        decode_fn=lambda s, sol: s.construct_full_path(sol),
        problem=test_problem,
    )
    print_algo_result("GA (GeneticAlgorithmSolver)", actions, cost, elapsed)

    # D) SA
    actions, cost, elapsed = run_solver(
        name="SA (SimulatedAnnealingSolver)",
        build_solver=lambda p: SimulatedAnnealingSolver(p),
        solve_fn=lambda s: s.solve(), 
        decode_fn=lambda s, sol: s.construct_full_path(sol),
        problem=test_problem,
    )
    print_algo_result("SA (SimulatedAnnealingSolver)", actions, cost, elapsed)

    # E) LS
    actions, cost, elapsed = run_solver(
        name="LS (LocalSearch)",
        build_solver=lambda p: LocalSearch(p),
        solve_fn=lambda s: s.solve(generations=50, verbose=False),
        decode_fn=lambda s, sol: s.construct_full_path(sol),
        problem=test_problem,
    )
    print_algo_result("LS (LocalSearch)", actions, cost, elapsed)
