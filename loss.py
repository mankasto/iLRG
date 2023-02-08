"""Define various loss functions and bundle them with appropriate metrics."""

import torch
import numpy as np


class Loss:
    """Abstract class, containing necessary methods.

    Abstract class to collect information about the 'higher-level' loss function, used to train an energy-based model
    containing the evaluation of the loss function, its gradients w.r.t. to first and second argument and evaluations
    of the actual metric that is targeted.

    """

    def __init__(self):
        """Init."""
        pass

    def __call__(self, reference, argmin):
        """Return l(x, y)."""
        raise NotImplementedError()
        return value, name, format

    def metric(self, reference, argmin):
        """The actually sought metric."""
        raise NotImplementedError()
        return value, name, format


class PSNR(Loss):
    """A classical MSE target.

    The minimized criterion is MSE Loss, the actual metric is average PSNR.
    """

    def __init__(self):
        """Init with torch MSE."""
        self.loss_fn = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    def __call__(self, x=None, y=None):
        """Return l(x, y)."""
        name = 'MSE'
        pf_fmt = '.6f'
        if x is None:
            return name, pf_fmt
        else:
            value = 0.5 * self.loss_fn(x, y)
            return value, name, pf_fmt

    def metric(self, x=None, y=None):
        """The actually sought metric."""
        name = 'avg PSNR'
        pf_fmt = '.3f'
        if x is None:
            return name, pf_fmt
        else:
            value = self.psnr_compute(x, y)
            return value, name, pf_fmt

    @staticmethod
    def psnr_compute(img_batch, ref_batch, batched=False, factor=1.0):
        """Standard PSNR."""

        def get_psnr(img_in, img_ref):
            mse = ((img_in - img_ref) ** 2).mean()
            if mse > 0 and torch.isfinite(mse):
                return (10 * torch.log10(factor ** 2 / mse)).item()
            elif not torch.isfinite(mse):
                return float('nan')
            else:
                return float('inf')

        if batched:
            psnr = get_psnr(img_batch.detach(), ref_batch)
        else:
            [B, C, m, n] = img_batch.shape
            psnrs = []
            for sample in range(B):
                psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
            psnr = np.mean(psnrs)

        return psnr


class Classification(Loss):
    """A classical NLL loss for classification. Evaluation has the softmax baked in.

    The minimized criterion is cross entropy, the actual metric is total accuracy.
    """

    def __init__(self):
        """Init with torch MSE."""
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                 reduce=None, reduction='mean')

    def __call__(self, x=None, y=None):
        """Return l(x, y)."""
        name = 'CrossEntropy'
        pf_fmt = '1.5f'
        if x is None:
            return name, pf_fmt
        else:
            value = self.loss_fn(x, y)
            return value, name, pf_fmt

    def metric(self, x=None, y=None):
        """The actually sought metric."""
        name = 'Accuracy'
        pf_fmt = '6.2%'
        if x is None:
            return name, pf_fmt
        else:
            value = (x.data.argmax(dim=1) == y).sum().float() / y.shape[0]
            return value.detach(), name, pf_fmt
