"""Define various loss functions and bundle them with appropriate metrics."""

import torch


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
