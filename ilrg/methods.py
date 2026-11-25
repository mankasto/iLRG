import torch
import numpy as np


# get_emb clips the decimal exponent gap between weight and bias gradients to this value
MAX_EXPONENT_GAP = 10
# Decimal exponent given to a zero bias gradient (with mantissa 1)
ZERO_BIAS_EXPONENT = -64


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


def get_emb(grad_w, grad_b, max_exponent_gap=MAX_EXPONENT_GAP):
    """Restore a class's average embedding as its weight-gradient row divided by its bias gradient.

    Mantissas and decimal exponents are divided separately, with the exponent gap clipped to
    +/- max_exponent_gap, so a tiny bias gradient cannot blow the result up.
    """
    sc_grad_b = '%e' % grad_b
    sc_grad_w = ['%e' % w for w in grad_w]
    real_b, exp_b = float(sc_grad_b.split('e')[0]), int(sc_grad_b.split('e')[1])
    real_w = np.array([float(sc_w.split('e')[0]) for sc_w in sc_grad_w])
    exp_w = np.array([int(sc_w.split('e')[1]) for sc_w in sc_grad_w])
    if real_b == 0.:
        real_b = 1
        exp_b = ZERO_BIAS_EXPONENT
    exp = exp_w - exp_b
    exp = np.where(exp > max_exponent_gap, max_exponent_gap, exp)
    exp = np.where(exp < -1 * max_exponent_gap, -1 * max_exponent_gap, exp)
    exp = np.array([10 ** x if x >= 0 else 1. / 10 ** (-x) for x in exp])
    res = (1. / real_b) * real_w * exp
    return torch.from_numpy(res).to(torch.float32)


def iLRG(probs, grad_b, n_classes, n_images):
    """Restore the number of instances of every class from the fc-layer bias gradient."""
    return _solve_label_counts(probs, grad_b, range(n_classes), n_images)


def sim_iLRG(probs, grad_b, exist_labels, n_images):
    """Restore the instance counts when the classes present in the batch are known."""
    return _solve_label_counts(probs, grad_b, exist_labels, n_images)


def _solve_label_counts(probs, grad_b, classes, n_images):
    """Solve iLRG's linear system for the instance count k_j of each class j in `classes`.

    The equations are sum_j k_j = n_images and, for each class i,
    sum_j k_j * (probs[j][i] - [i == j]) = n_images * grad_b[i], where probs[j] are the
    restored average probabilities of class j. They are solved in the least-squares sense.
    """
    coefs, values = [[1 for _ in classes]], [n_images]
    for i in classes:
        coefs.append([probs[j][i].item() - 1 if j == i else probs[j][i].item() for j in classes])
        values.append(n_images * grad_b[i])
    res_float = np.linalg.pinv(np.array(coefs)).dot(np.array(values))
    return _round_counts(res_float, n_images)


def _round_counts(res_float, n_images):
    """Round the least-squares counts (negative ones and ones above n_images become 0).

    Returns the rounded counts and a copy adjusted by one per class, chosen by rounding error,
    towards a total of n_images.
    """
    res = np.where(res_float > 0, res_float, 0)
    res = np.round(res).astype(int)
    res = np.where(res <= n_images, res, 0)
    err = res - res_float
    num_mod = np.sum(res) - n_images
    if num_mod > 0:
        mod_res = res.copy()
        mod_res[np.argsort(-err)[:num_mod]] -= 1
    elif num_mod < 0:
        mod_res = res.copy()
        mod_res[np.argsort(err)[:num_mod]] += 1
    else:
        mod_res = res
    return res, mod_res
