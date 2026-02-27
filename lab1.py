import random as rnd
from ast import Return

import numpy as np
from numpy._core.umath import matmul, vecdot
from numpy.ma.core import mean
from numpy.matrixlib.defmatrix import matrix


def task_one():
    return {
        "array": np.zeros(10),
    }


def task_two():
    arr = np.zeros(10)
    arr[4] = 1
    return {
        "array": arr,
    }


def task_three():
    arr = np.array([rnd.randint(0, 100) for _ in range(20)], dtype=np.int32)
    return {
        "array": arr,
        "nonzero_indices": np.nonzero(arr)[0],
    }


def task_four():
    tensor_int = np.random.randint(0, 10, (3, 3, 3))
    return {
        "tensor_int": tensor_int,
    }


def task_five():
    arr = np.random.randint(0, 100, (10))
    return {
        "array": arr,
        "mean": np.mean(arr),
    }


def task_six():
    matrix_1 = np.random.randint(0, 100, (5, 3))
    matrix_2 = np.random.randint(0, 100, (3, 2))
    return {
        "matrix_1": matrix_1,
        "matrix_2": matrix_2,
        "multiplication_result": matrix_1 @ matrix_2,
    }


def task_seven():
    matrix_1 = np.random.randint(0, 100, (4, 4))
    matrix_2 = np.random.randint(0, 100, (4, 4))
    mult = matrix_1 @ matrix_2
    return {
        "matrix_1": matrix_1,
        "matrix_2": matrix_2,
        "multiplication_result": mult,
        "diagonal": np.diagonal(mult),
    }


def task_eight():
    vec = np.random.randint(0, 100, (20))
    vec[vec.argmax()] = 0
    return {
        "vector": vec,
    }


def task_nine():
    vec = np.random.randint(0, 100, (10))
    return {
        "vector": vec,
        "unique_elements": np.unique(vec),
    }


def task_ten():
    matrix = np.random.randint(0, 100, (3, 3))
    res = (matrix.T - matrix.mean(axis=1)).T
    return {
        "matrix": matrix,
        "mean": np.mean(matrix),
        "result": res,
    }


def task_eleven():
    matrix = np.random.randint(0, 100, (3, 3))
    matrix_before = matrix.copy()
    tmp = matrix[0].copy()
    matrix[0] = matrix[1]
    matrix[1] = tmp

    return {
        "matrix_before": matrix_before,
        "matrix": matrix,
        "mean": np.mean(matrix),
    }


def task_twelve(n=5):
    vec = np.random.randint(0, 100, (20))
    return {
        "vector": vec,
        "top_n": np.sort(vec)[-n:],
    }


def task_thirteen():
    matrix = np.random.randint(0, 10, (5, 5))
    return {
        "matrix": matrix,
        "sum": np.sum(matrix, axis=1),
    }


def task_fourteen():
    vec = np.random.randint(-1, 1, (10))
    new_vec = vec.copy()
    new_vec[new_vec == -1] = 1
    return {
        "vector": vec,
        "new_vector": new_vec,
    }


def task_fifteen():
    vec = np.random.randint(1, 100, (12))
    parts = np.split(vec, 3)
    sums = [np.sum(part) for part in parts]
    return {
        "vector": vec,
        "parts": parts,
        "sums": sums,
    }
