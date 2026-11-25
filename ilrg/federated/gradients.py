"""Gradients a client shares: the fc-layer (or all) parameter gradients, and their degraded
versions under a defense."""

import torch


def get_grads(outs, labels, model, loss_fn, rec=False):
    loss = loss_fn(outs, labels)
    model.zero_grad()
    grads = torch.autograd.grad(loss, model.parameters()) if rec else torch.autograd.grad(loss, model.fc.parameters())
    grads = list((_.detach().cpu().clone() for _ in grads))
    return grads


def degrade_grads(grads,
                  defense_method='dp',
                  model_name='resnet18',
                  param_value=0.001,
                  images=None,
                  model=None):
    from .defense import additive_noise, gradient_clipping, \
        gradient_compression, perturb_representation
    if defense_method == 'dp':
        lossy_grads = additive_noise(grads, param_value)
    elif defense_method == 'clip':
        lossy_grads = gradient_clipping(grads, param_value)
    elif defense_method == 'sparse':
        lossy_grads = gradient_compression(grads, param_value)
    elif defense_method == 'perturb':
        assert model_name == 'resnet18', 'Soteria Perturbation only applicable to resnet18'
        lossy_grads = perturb_representation(grads, model, images, param_value)
    return lossy_grads
