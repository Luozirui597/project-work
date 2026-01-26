from Problem import Problem
from src.SplitGA import SplitGASolver


def solution(p: Problem):
    POP_SIZE = 50
    GENERATIONS = 100
    MUTATION_RATE = 0.2

    solver = SplitGASolver(p)
    best_individual = solver.solve(
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
    )

    path = solver.construct_full_path(best_individual)

    if not path:
        return [(0, 0)]
    last_city, last_gold = path[-1]
    if int(last_city) != 0 or float(last_gold) != 0.0:
        path.append((0, 0))
    else:
        path[-1] = (0, 0)

    return path
