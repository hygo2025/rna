import sys
import os
from typing import List, Dict, Set


def build_graph(n: int, portals: List[int]) -> Dict[int, Set[int]]:
    adj = {}
    for i in range(1, n):
        portal = portals[i - 1]
        target = i + portal
        adj[i] = target

    return adj

def run_simulation(start_from: int, to_cell: int, adj: Dict[int, Set[int]]) -> bool:
    current_cell = start_from

    while current_cell < to_cell and current_cell in adj:
        current_cell = adj[current_cell]

    return current_cell == to_cell

def solve():
    try:
        n_str, t_str = sys.stdin.readline().split()
        n = int(n_str)
        t = int(t_str)

        portals = [int(p) for p in sys.stdin.readline().split()]

    except (IOError, ValueError):
        return

    result = run_simulation(
        start_from=1,
        to_cell=t,
        adj=build_graph(n, portals),
    )

    if result:
        print("YES")
    else:
        print("NO")


if __name__ == "__main__":
    input_filename = "input.txt"
    if os.path.exists(input_filename):
        sys.stdin = open(input_filename, 'r')
    else:
        print("--- Arquivo 'input.txt' não encontrado. Aguardando entrada manual... ---")

    solve()