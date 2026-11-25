"""iLRG: class-wise embeddings and probabilities from the fc-layer gradients, then the number of
instances of every class."""

import torch
import numpy as np

# get_emb clips the decimal exponent gap between weight and bias gradients to this value
MAX_EXPONENT_GAP = 10

# Decimal exponent given to a zero bias gradient (with mantissa 1)
ZERO_BIAS_EXPONENT = -64


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


def post_process_emb(embedding, model, device, alpha=0.01):
    embedding = embedding.to(device)
    # Feed embedding into FC-Layer to get probabilities.
    out = model.fc(embedding) * alpha
    prob = torch.softmax(out, dim=-1)
    return prob


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
