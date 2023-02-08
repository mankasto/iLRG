import torch
import numpy as np


def solve_lp(A, b, c):
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
    np.random.seed(None)
    for _ in range(1):
        A, b, c = matrix(A), matrix(b), matrix(c)
        sol = solvers.lp(c, A, b)
        x = sol['x']
        if x is not None:
            ret = A * x
            if ret[0] < -0.1 and np.max(ret[1:]) < 1e-2 and np.count_nonzero(np.array(ret[1:]) <= 0) > 0.5 * len(ret):
                return True
    return False


def solve_perceptron(X, y, fit_intercept=True, max_iter=1000, tol=1e-3, eta0=1.):
    from sklearn.linear_model import Perceptron
    clf = Perceptron(fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, eta0=eta0)
    clf.fit(X, y)
    if not fit_intercept:
        pass
    if clf.score(X, y) > 0.9:
        return True
    return False


def svd_infer(A, num_classes=1000, gt_k=None, epsilon=1e-8):
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


# Extended GradientInversion, w_grad: m * n.
def gi_infer(w_grad):
    ret = (torch.min(w_grad, dim=0)[0] < 0).nonzero(as_tuple=False).squeeze().numpy()
    ret = [ret] if ret.shape == () else ret
    return ret


# Extended iDLG
def idlg_infer(w_grad):
    ret = (torch.sum(w_grad, dim=0) < 0).nonzero(as_tuple=False).squeeze().numpy()
    ret = [ret] if ret.shape == () else ret
    return ret


# Recover embeddings
def get_emb(grad_w, grad_b, exp_thre=10):
    # Split scientific count notation
    sc_grad_b = '%e' % grad_b
    sc_grad_w = ['%e' % w for w in grad_w]
    real_b, exp_b = float(sc_grad_b.split('e')[0]), int(sc_grad_b.split('e')[1])
    real_w, exp_w = np.array([float(sc_w.split('e')[0]) for sc_w in sc_grad_w]), \
                    np.array([int(sc_w.split('e')[1]) for sc_w in sc_grad_w])
    # Deal with 0 case
    if real_b == 0.:
        real_b = 1
        exp_b = -64
    # Deal with exponent value
    exp = exp_w - exp_b
    exp = np.where(exp > exp_thre, exp_thre, exp)
    exp = np.where(exp < -1 * exp_thre, -1 * exp_thre, exp)

    def get_exp(x):
        return 10 ** x if x >= 0 else 1. / 10 ** (-x)

    exp = np.array(list(map(get_exp, exp)))
    # Calculate recovered average embeddings for batch_i (samples of class i)
    res = (1. / real_b) * real_w * exp
    res = torch.from_numpy(res).to(torch.float32)
    return res


# Recover Labels
def iLRG(probs, grad_b, n_classes, n_images):
    # Solve linear equations to recover labels
    coefs, values = [], []
    # Add the first equation: k1+k2+...+kc=K
    coefs.append([1 for _ in range(n_classes)])
    values.append(n_images)
    # Add the following equations
    for i in range(n_classes):
        coef = []
        for j in range(n_classes):
            if j != i:
                coef.append(probs[j][i].item())
            else:
                coef.append(probs[j][i].item() - 1)
        coefs.append(coef)
        values.append(n_images * grad_b[i])
    # Convert into numpy ndarray
    coefs = np.array(coefs)
    values = np.array(values)
    # Solve with Moore-Penrose pseudoinverse
    res_float = np.linalg.pinv(coefs).dot(values)
    # Filter negative values
    res = np.where(res_float > 0, res_float, 0)
    # Round values
    res = np.round(res).astype(int)
    res = np.where(res <= n_images, res, 0)
    err = res - res_float
    num_mod = np.sum(res) - n_images
    if num_mod > 0:
        inds = np.argsort(-err)
        mod_inds = inds[:num_mod]
        mod_res = res.copy()
        mod_res[mod_inds] -= 1
    elif num_mod < 0:
        inds = np.argsort(err)
        mod_inds = inds[:num_mod]
        mod_res = res.copy()
        mod_res[mod_inds] += 1
    else:
        mod_res = res

    return res, mod_res


# Have Known about which labels exist
def sim_iLRG(probs, grad_b, exist_labels, n_images):
    # Solve linear equations to recover labels
    coefs, values = [], []
    # Add the first equation: k1+k2+...+kc=K
    coefs.append([1 for _ in range(len(exist_labels))])
    values.append(n_images)
    # Add the following equations
    for i in exist_labels:
        coef = []
        for j in exist_labels:
            if j != i:
                coef.append(probs[j][i].item())
            else:
                coef.append(probs[j][i].item() - 1)
        coefs.append(coef)
        values.append(n_images * grad_b[i])
    # Convert into numpy ndarray
    coefs = np.array(coefs)
    values = np.array(values)
    # Solve with Moore-Penrose pseudoinverse
    res_float = np.linalg.pinv(coefs).dot(values)
    # Filter negative values
    res = np.where(res_float > 0, res_float, 0)
    # Round values
    res = np.round(res).astype(int)
    res = np.where(res <= n_images, res, 0)
    err = res - res_float
    num_mod = np.sum(res) - n_images
    if num_mod > 0:
        inds = np.argsort(-err)
        mod_inds = inds[:num_mod]
        mod_res = res.copy()
        mod_res[mod_inds] -= 1
    elif num_mod < 0:
        inds = np.argsort(err)
        mod_inds = inds[:num_mod]
        mod_res = res.copy()
        mod_res[mod_inds] += 1
    else:
        mod_res = res

    return res, mod_res
