"""
Перемножение матриц с использованием Pool (пула процессов).

Pool автоматически распределяет задачи между фиксированным числом процессов.
Это эффективнее, чем создавать отдельный процесс на каждый элемент.

Задания:
  TODO 3 — использовать Pool.starmap() для параллельного вычисления
  TODO 4 — сравнить время при разном числе процессов в пуле

Запуск:
    python3 03_pool_matrix.py

═══════════════════════════════════════════════════════════════════════
СПРАВКА: Зачем нужен Pool (из репозитория 3_Parallelism)
https://github.com/fa-python-network/3_Parallelism
═══════════════════════════════════════════════════════════════════════

В файле 02_matrix_multiply.py для каждого элемента матрицы создавался
отдельный Process. При матрице 50x50 это 2500 процессов — крайне
неэффективно, так как на создание каждого процесса тратится время.

Pool решает эту проблему: создаётся фиксированное число процессов
(обычно = количеству ядер CPU), и задачи распределяются между ними.

Задание из репозитория:
  «Используйте пул процессов, чтобы распределять вычисления между
   определённым заранее количеством процессов, не зависящим от размеров
   матрицы.»

Именно это вы реализуете ниже с помощью Pool.starmap().
═══════════════════════════════════════════════════════════════════════
"""

import time
import os
from multiprocessing import Pool


def element(i, j, A, B):
    N = len(A[0])
    res = 0
    for k in range(N):
        res += A[i][k] * B[k][j]
    return (i, j, res)


SIZE = 50

matrix_a = [[(i + j) % 10 for j in range(SIZE)] for i in range(SIZE)]
matrix_b = [[(i * j) % 10 for j in range(SIZE)] for i in range(SIZE)]


def sequential_multiply(A, B):
    rows = len(A)
    cols = len(B[0])
    result = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            _, _, val = element(i, j, A, B)
            result[i][j] = val
    return result


def pool_multiply(A, B, num_processes):
    rows = len(A)
    cols = len(B[0])
    result = [[0] * cols for _ in range(rows)]

    args = [(i, j, A, B) for i in range(rows) for j in range(cols)]

    with Pool(processes=num_processes) as pool:
        results_list = pool.starmap(element, args)

    for i, j, val in results_list:
        result[i][j] = val

    return result


if __name__ == '__main__':
    cpu_count = os.cpu_count()
    print(f"Размер матриц: {SIZE}x{SIZE}")
    print(f"Доступно ядер CPU: {cpu_count}\n")

    t = time.time()
    seq_result = sequential_multiply(matrix_a, matrix_b)
    time_seq = time.time() - t
    print(f"Последовательно: {time_seq:.4f} сек")

    for n in [1, 2, 4]:
        t = time.time()
        par_result = pool_multiply(matrix_a, matrix_b, n)
        elapsed = time.time() - t
        print(f"Pool ({n} процессов): {elapsed:.4f} сек")
        assert par_result == seq_result, "Результаты не совпадают!"