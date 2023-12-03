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





m = [0.0 for i in range(196)]
m = np.array(m).reshape(14, 14)
m[0][12] = 1/3
m[1][1], m[1][2] = 1/2, 1/6
m[2][2], m[2][1], m[2][3] = 1/6, 1/2, 1/4
m[3][2]=1/6
m[4][5]=1/3
m[5][5], m[5][6]=1/3, 1/3
m[6][2]=1/6
m[7][6],m[7][3], m[7][0]=1/3, 1/4, 1/4
m[8][4]=1
m[9][0]=1/4
m[10][2], m[10][6], m[10][5], m[10][12]=1/6, 1/3, 1/3, 1/3
m[11][3],m[11][0]=1/4, 1/4
m[13][0], m[13][12], m[13][3], m[13][2]= 1/4, 1/3, 1/4, 1/6
np.set_printoptions(precision=2, suppress=True)
print(m)
eigenvalues, eigenvectors = np.linalg.eig(m)
print(eigenvalues)
print(eigenvectors)
v = pagerank(m, 5, 1)
print(v)
