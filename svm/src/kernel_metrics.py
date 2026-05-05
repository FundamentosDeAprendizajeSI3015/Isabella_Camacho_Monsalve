import numpy as np

def complex_ratio(A):
    """Computes the trace of a matrix."""
    return np.trace(A)

def FSM(K, y):
    """Computes the Feature Space Measure (FSM)."""
    n_nega = np.count_nonzero(y == -1)  # Number of negative class samples
    n_posi = len(y) - n_nega  # Number of positive class samples
    if n_nega <= 1 or n_posi <= 1:
        return 0.0

    # Compute intra-class and inter-class similarities
    d_i = np.sum(K[np.ix_(y == -1, y == -1)], axis=1) / n_nega  # Negative class intra-similarity
    a_i = np.sum(K[np.ix_(y == 1, y == 1)], axis=1) / n_posi   # Positive class intra-similarity
    c_i = np.sum(K[np.ix_(y == -1, y == 1)], axis=1) / n_posi  # Negative-positive inter-similarity
    b_i = np.sum(K[np.ix_(y == 1, y == -1)], axis=1) / n_nega  # Positive-negative inter-similarity

    # Compute global similarity measures
    A = sum(a_i) / n_posi
    B = sum(b_i) / n_posi
    C = sum(c_i) / n_nega
    D = sum(d_i) / n_nega

    rest_phi_square = A + D - B - C  # Normalization factor
    if np.isclose(rest_phi_square, 0.0):
        return 0.0

    aux_1 = np.true_divide(sum(np.square(b_i - a_i + A - B)), rest_phi_square * (n_posi - 1))
    aux_2 = np.true_divide(sum(np.square(c_i - d_i + D - C)), rest_phi_square * (n_nega - 1))

    return np.true_divide(np.sqrt(aux_1) + np.sqrt(aux_2), np.sqrt(rest_phi_square))

def ideal_kernel(y):
    """Constructs the ideal kernel based on labels."""
    K_ideal = np.equal.outer(y, y).astype(int)  # Creates a similarity matrix
    K_ideal = np.where(K_ideal == 0, -1 + K_ideal, K_ideal)  # Assigns -1 to different class pairs
    return K_ideal

def kernel_alignment(K, y):
    """Computes the kernel alignment."""
    A1 = np.trace(np.dot(K.transpose(), ideal_kernel(y)))  # Inner product with ideal kernel
    denom = np.linalg.norm(K, 'fro') * len(y)
    if np.isclose(denom, 0.0):
        return 0.0
    return A1 / denom  # Normalized alignment score


def kernel_aligment(K, y):
    """Backward-compatible misspelled alias for kernel_alignment."""
    return kernel_alignment(K, y)

def kernel_polarization(k, y):
    """Computes kernel polarization."""
    y = np.asarray(y)
    diag = np.diag(k)
    pairwise = diag[:, None] + diag[None, :] - 2 * k
    label_outer = np.outer(y, y)
    polarization = -label_outer * pairwise
    np.fill_diagonal(polarization, 0.0)
    return float(np.sum(polarization))
