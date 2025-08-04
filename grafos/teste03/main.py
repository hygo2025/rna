import sys
import os
import heapq
from typing import List, Dict, Tuple

# https://vjudge.net/contest/727919#problem/A

def get_valid_knight_moves(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    l, c = pos

    possible_moves = [(l + 2, c + 1), (l + 2, c - 1), (l - 2, c + 1), (l - 2, c - 1),
        (l + 1, c + 2), (l + 1, c - 2), (l - 1, c + 2), (l - 1, c - 2),
    ]

    valid_moves = []
    for new_r, new_c in possible_moves:
        if 0 <= new_r <= 7 and 0 <= new_c <= 7:
            valid_moves.append((new_r, new_c))

    return valid_moves


def find_minimum_cost(start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> int:
    distances: Dict[Tuple[int, int], int] = {}
    pq: List[Tuple[int, Tuple[int, int]]] = [(0, start_pos)]
    distances[start_pos] = 0

    while pq:
        current_cost, current_pos = heapq.heappop(pq)

        if current_cost > distances.get(current_pos, float('inf')):
            continue

        if current_pos == end_pos:
            return current_cost

        for next_pos in get_valid_knight_moves(current_pos):
            r1, c1 = current_pos
            r2, c2 = next_pos
            move_cost = (r1 * r2) + (c1 * c2)
            new_total_cost = current_cost + move_cost

            if new_total_cost < distances.get(next_pos, float('inf')):
                distances[next_pos] = new_total_cost
                heapq.heappush(pq, (new_total_cost, next_pos))

    return -1


def solve():
    try:
        all_lines = sys.stdin.readlines()
    except IOError:
        all_lines = []

    for line in all_lines:
        parts = [int(p) for p in line.split()]
        if len(parts) < 4: continue

        start_pos: Tuple[int, int] = (parts[0], parts[1])
        end_pos: Tuple[int, int] = (parts[2], parts[3])

        min_cost = find_minimum_cost(start_pos, end_pos)
        print(min_cost)

#solve()

def display_board():
    print("-------------------------------------------------")
    print("       | (c-2) | (c-1) |  (c)  | (c+1) | (c+2) |")
    print("-------+-------+-------+-------+-------+-------+")
    print("(r+2)  |   .   |   7   |   .   |   8   |   .   |")
    print("(r+1)  |   5   |   .   |   .   |   6   |   .   |")
    print("(r)    |   .   |   .   |   C   |   .   |   .   |")
    print("(r-1)  |   3   |   .   |   .   |   4   |   .   |")
    print("(r-2)  |   .   |   1   |   .   |   2   |   .   |")
    print("-------------------------------------------------")


if __name__ == "__main__":
    display_board()

    input_filename = "input.txt"
    if os.path.exists(input_filename):
        print(f"\nRunning test cases from file '{input_filename}':\n")
        sys.stdin = open(input_filename, 'r')
        solve()
    else:
        pass