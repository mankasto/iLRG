"""Label restoration baselines compared with iLRG: the SVD method, extended GradInversion and
extended iDLG."""

import torch
import numpy as np


def solve_lp(A, b, c):
    """SVD baseline: whether the LP min c'x subject to A x <= b separates column 0 from the others."""
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
    np.random.seed(None)
    A, b, c = matrix(A), matrix(b), matrix(c)
    x = solvers.lp(c, A, b)['x']
    if x is None:
        return False
    ret = A * x
    if ret[0] < -0.1 and np.max(ret[1:]) < 1e-2 and np.count_nonzero(np.array(ret[1:]) <= 0) > 0.5 * len(ret):
        return True
    return False


def solve_perceptron(X, y, fit_intercept=True, max_iter=1000, tol=1e-3, eta0=1.):
    """SVD baseline: whether a perceptron separates the samples X by the labels y (score above 0.9)."""
    from sklearn.linear_model import Perceptron
    clf = Perceptron(fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, eta0=eta0)
    clf.fit(X, y)
    if clf.score(X, y) > 0.9:
        return True
    return False


def svd_infer(A, num_classes=1000, gt_k=None, epsilon=1e-8):
    """SVD baseline: classes whose coordinates in the top singular subspace of A (m x n) pass a
    perceptron and a linear-programming separability test."""
    m, n = np.shape(A)
    B, s, C = np.linalg.svd(A, full_matrices=False)
    pred_k = np.linalg.matrix_rank(A)
    k = min(gt_k, pred_k)
    C = C[:k, :].astype(np.double)
    # Find x: x @ C has only one positive element
    # Filter possible labels using perceptron algorithm
    bow = []
    for i in range(n):
        if i in bow:
            continue
        indices = [j for j in range(n) if j != i]
        np.random.shuffle(indices)
        if solve_perceptron(
                X=np.concatenate([C[:, i:i + 1], C[:, indices[:num_classes - 1]]], 1).transpose(),
                y=np.array([1 if j == 0 else -1 for j in range(num_classes)]),
                fit_intercept=True,
                max_iter=1000,
                tol=1e-3
        ):
            bow.append(i)
    # Get the final set with linear programming
    ret_bow = []
    for i in bow:
        if i in ret_bow:
            continue
        indices = [j for j in range(n) if j != i]
        D = np.concatenate([C[:, i:i + 1], C[:, indices]], 1)
        indices2 = np.argsort(np.linalg.norm(D[:, 1:], axis=0))
        A = np.concatenate([D[:, 0:1], -D[:, 1 + indices2]], 1).transpose()
        if solve_lp(
                A=A,
                b=np.array([-epsilon] + [0] * len(indices2)),
                c=np.array(C[:, i:i + 1])
        ):
            ret_bow.append(i)
    return ret_bow


def gi_infer(w_grad):
    """Extended GradInversion: classes with a negative entry in their column of w_grad (m x n)."""
    ret = (torch.min(w_grad, dim=0)[0] < 0).nonzero(as_tuple=False).squeeze().numpy()
    ret = [ret] if ret.shape == () else ret
    return ret


def idlg_infer(w_grad):
    """Extended iDLG: classes whose column of w_grad (m x n) sums to a negative value."""
    ret = (torch.sum(w_grad, dim=0) < 0).nonzero(as_tuple=False).squeeze().numpy()
    ret = [ret] if ret.shape == () else ret
    return ret
