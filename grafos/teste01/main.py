import sys
import os
from typing import List, Dict, Set, Tuple


def build_graph_from_lines(n: int, edge_lines: List[str]) -> Tuple[Dict[int, Set[int]], List[int]]:
    # inicializa geral sem vizinhos
    adj = {i: set() for i in range(1, n + 1)}
    degrees = [0] * n

    for line in edge_lines:
        a_str, b_str = line.split()
        a, b = int(a_str), int(b_str)

        # no enunciado do problema diz que a liga b e vice versa, se ja existe uma ligacao eu nao devo add novamente
        if b not in adj[a]:
            adj[a].add(b)
            adj[b].add(a)
            degrees[a - 1] += 1
            degrees[b - 1] += 1

    return adj, degrees


def run_simulation(n: int, adj: Dict[int, Set[int]], degrees: List[int]) -> int:
    counter = 0

    while True:
        to_remove = []
        for i in range(n):
            if degrees[i] == 1: # se tiver ligado a somente mais um aluno marca para remover
                to_remove.append(i + 1)

        if not to_remove:
            break

        counter += 1

        for elem in to_remove: # eu removo todo mundo que eu achei acima
            degrees[elem - 1] = 0

        for elem in to_remove:
            for idx_remove in adj[elem]: # se 1 é vizinho de 2 e eu tirei 1 acima, eu preciso diminuir o grau de 2 visto que removi acima
                if degrees[idx_remove - 1] > 0:
                    degrees[idx_remove - 1] -= 1

    return counter


def solve():
    lines = sys.stdin.readlines()
    if not lines:
        return

    try:
        first_line = lines[0]
        n_str, m_str = first_line.split()
        n = int(n_str)
        m = int(m_str)
    except (IOError, ValueError, IndexError):
        return

    if n == 0:
        print(0)
        return

    edges = lines[1: m + 1]
    adj, degrees = build_graph_from_lines(n, edges)
    result = run_simulation(n, adj, degrees)
    print(result)


if __name__ == "__main__":
    input_filename = "input.txt"
    if os.path.exists(input_filename):
        sys.stdin = open(input_filename, 'r')
    else:
        print("--- Arquivo 'input.txt' não encontrado. Aguardando entrada manual... ---")

    solve()