import math
import numpy as np


def pagerank(M, num_iterations: int = 100, d: float = 0.85):
    """PageRank algorithm with explicit number of iterations. Returns ranking of nodes (pages) in the adjacency matrix.

    Parameters
    ----------
    M : numpy array
        adjacency matrix where M_i,j represents the link from 'j' to 'i', such that for all 'j'
        sum(i, M_i,j) = 1
    num_iterations : int, optional
        number of iterations, by default 100
    d : float, optional
        damping factor, by default 0.85

    Returns
    -------
    numpy array
        a vector of ranks such that v_i is the i-th rank from [0, 1],
        v sums to 1

    """
    N = M.shape[1]
    v = np.ones(N) / N
    M_hat = (d * M + (1 - d) / N)
    for i in range(num_iterations):
        v = M_hat @ v
    return v


m = [0.0 for i in range(100)]
m = np.array(m).reshape(10, 10)
m[0][0], m[0][1] = 1 / 5, 1 / 4
m[1][0], m[1][4] = 1 / 5, 1 / 2
m[2][5], m[2][8] = 1, 1 / 3
m[3][0] = 1 / 5
m[4][1], m[4][2], m[4][3], m[4][7] = 1 / 4, 1 / 4, 1 / 2, 1
m[5][2], m[5][9] = 1 / 4, 1 / 2
m[6][0], m[6][3], m[6][8] = 1 / 5, 1 / 2, 1 / 3
m[7][1], m[7][2], m[7][4], m[7][9] = 1 / 4, 1 / 4, 1 / 2, 1 / 2
m[8][6] = 1
m[9][8], m[9][1], m[9][0], m[9][2] = 1 / 3, 1 / 4, 1 / 5, 1 / 4
m = m
np.set_printoptions(precision=2, suppress=True)
print(m)
eigenvalues, eigenvectors = np.linalg.eig(m)
print(eigenvalues)
print(eigenvectors)
v = pagerank(m, 999999, 1)
print(v)
vec = [
    0.696,
    2.226,
    0.804,
    0.139,
    4.173,
    0.701,
    0.312,
    3.345,
    0.312,
    1
]
sum = 0
for i in vec:
    sum += (i * i)
vec1 = list()
print(math.sqrt(sum))
for i in vec:
    vec1.append(i / (sum) ** 0.5)
vec1 = np.array(vec1)
print(f'vec {vec}')
print(f'vec {vec1}')
vec2 = list()
sum = 0
for i in vec1:
    sum += (i)
for i in vec1:
    vec2.append(i / (sum))
print(sum)
vec2 = np.array(vec2)
print(f'vec {vec2}')
