import os
import sys
from typing import List, Tuple

# Add repo root to sys.path so we can import Problem and s328249
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Problem import Problem
from s328249 import solution


def compute_total_cost(p: Problem, actions: List[Tuple[int, float]]) -> float:
    """
    Compute total cost of an action list [(city, gold), ...] using Problem.cost().
    Rule assumed (consistent with your earlier convention):
      - You travel from actions[i].city -> actions[i+1].city with current carried_gold
      - Upon arriving at the next city, you collect actions[i+1].gold (if > 0)
      - When you arrive at city 0, you unload (carried_gold = 0)
    """
    if not actions:
        return float("inf")

    total = 0.0
    carried = 0.0

    # start at first node in actions
    prev_city = int(actions[0][0])

    # collect gold at starting city if any (usually 0)
    start_gold = float(actions[0][1])
    if prev_city == 0:
        carried = 0.0
    if start_gold > 0:
        carried += start_gold

    for city, gold in actions[1:]:
        city = int(city)
        gold = float(gold)

        # travel prev_city -> city with current carried weight
        total += p.cost([prev_city, city], carried)

        # arrive: collect
        if gold > 0:
            carried += gold

        # unload at depot
        if city == 0:
            carried = 0.0

        prev_city = city

    return float(total)


def main():
    # You can change parameters here
    p = Problem(num_cities=100, alpha=1, beta=2, density=0.5)

    # Baseline (provided by Problem.py) returns a total cost (float)
    baseline_cost = p.baseline()

    # Your solver path
    path = solution(p)

    # Print full path (as you requested)
    print("PATH:")
    print(path)

    # Compute cost of your path
    my_cost = compute_total_cost(p, path)

    # Compare with baseline
    # Improvement: positive means better (lower cost than baseline)
    if baseline_cost > 0:
        improvement_pct = (baseline_cost - my_cost) / baseline_cost * 100.0
    else:
        improvement_pct = float("nan")

    print("\nCOSTS:")
    print(f"My path cost:      {my_cost:.6f}")
    print(f"Baseline cost:     {baseline_cost:.6f}")
    print(f"Improvement vs BL: {improvement_pct:+.3f}%")

    # Optional: quick sanity checks
    if path:
        print("\nSANITY:")
        print(f"First step: {path[0]}")
        print(f"Last step:  {path[-1]}")


if __name__ == "__main__":
    main()
