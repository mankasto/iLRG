"""Mechanisms for image reconstruction from parameter gradients."""

import os
import time
import torch
import torchvision

from .utils import get_target_data, lr_milestones
from collections import defaultdict
from .medianfilt import MedianPool2d
from .metrics import InceptionScore
from .metrics import total_variation as TV

SGD_LR = 0.01
SGD_MOMENTUM = 0.9
MEDIAN_FILTER_SIZE = 3
# --indices choices: how many parameter gradients enter the cost
TOP_NORM_COUNTS = {'topk-1': 4, 'top10': 10, 'top20': 20, 'top30': 30, 'top50': 50}
FIRST_COUNTS = {'first': 4, 'first4': 4, 'first5': 5, 'first10': 10, 'first50': 50}
LAST_COUNTS = {'last5': 5, 'last10': 10, 'last50': 50}
RANDOM_SUBSET_SIZE = 8


class GradientReconstructor:
    """Instantiate a reconstruction algorithm."""

    def __init__(self, model, config, rec_exp_dir, mean_std=(0.0, 1.0), num_images=1, loss_thresh=1e-4):
        """Initialize with algorithm setup."""
        self.config = config
        self.model = model
        self.setup = dict(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
        self.mean_std = mean_std
        self.num_images = num_images
        self.reconstruct_label = True
        self.early_stop = True
        self.loss_thresh = loss_thresh
        self.exp_dir = rec_exp_dir

        if self.config['scoring_choice'] == 'inception':
            self.inception = InceptionScore(batch_size=1, setup=self.setup)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    def reconstruct(self, input_data, labels, img_shape=(3, 32, 32), dryrun=False, eval=True, tol=None,
                    aux_data=None):
        """Reconstruct image from gradient."""
        start_time = time.time()
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        if eval:
            self.model.eval()

        stats = defaultdict(list)
        x = self._init_images(img_shape, labels, aux_data)
        scores = torch.zeros(self.config['restarts'])

        if labels is None:
            def loss_fn(pred, label):
                label = torch.nn.functional.softmax(label, dim=-1)
                return torch.mean(torch.sum(- label * torch.nn.functional.log_softmax(pred, dim=-1), 1))

            self.loss_fn = loss_fn
        else:
            assert labels.shape[0] == self.num_images
            self.reconstruct_label = False

        try:
            for trial in range(self.config['restarts']):
                x_trial, labels = self._run_trial(x[trial], input_data, labels, dryrun=dryrun)
                # Finalize
                scores[trial] = self._score_trial(x_trial, input_data, labels)
                x[trial] = x_trial
                if tol is not None and scores[trial] <= tol:
                    break
                if dryrun:
                    break
        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass

        # Choose optimal result:
        if self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            x_optimal, stats = self._average_trials(x, labels, input_data, stats)
        else:
            print('Choosing optimal result ...')
            scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
            optimal_index = torch.argmin(scores)
            print(f'Optimal result score: {scores[optimal_index]:2.4f}')
            stats['opt'] = scores[optimal_index].item()
            x_optimal = x[optimal_index]

        print(f'Total time: {time.time() - start_time}.')
        return x_optimal.detach(), stats

    def _init_images(self, img_shape, labels=None, aux_data=None):
        if self.config['init'] == 'randn':
            return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'rand':
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'auxiliary':
            assert aux_data is not None, 'Please provide an auxiliary dataset'
            target_id = 0
            images = []
            for _ in range(self.config['restarts']):
                print('get data')
                image, target_id = get_target_data(aux_data, labels, target_id,
                                                   device=self.setup['device'])
                images.append(image)
            print('finish init')
            return images
        else:
            raise ValueError()

    def _run_trial(self, x_trial, input_data, labels, dryrun=False):
        x_trial.requires_grad = True
        if self.reconstruct_label:
            output_test = self.model(x_trial)[0]
            labels = torch.randn(output_test.shape[1]).to(**self.setup).requires_grad_(True)
            optimizer = self._optimizer([x_trial, labels])
        else:
            optimizer = self._optimizer([x_trial])

        max_iterations = self.config['max_iterations']
        dm, ds = self.mean_std
        if self.config['lr_decay']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones(max_iterations),
                                                             gamma=0.1)
        try:
            if self.config['init'] == 'auxiliary':
                self._filter(x_trial)
                aux_den = torch.clamp(x_trial * ds + dm, 0, 1)
                for j in range(self.num_images):
                    filename = f'aux_{j}.png'
                    torchvision.utils.save_image(aux_den[j:j + 1, ...],
                                                 os.path.join(self.exp_dir, filename))
            print('start iteration steps')
            start_iteration = 0
            for iteration in range(start_iteration, max_iterations):
                log_num = 100
                save_interval = 500
                closure = self._gradient_closure(optimizer, x_trial, input_data, labels)
                rec_loss = optimizer.step(closure)
                if self.config['lr_decay']:
                    scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.config['boxed']:
                        x_trial.data = torch.max(torch.min(x_trial, (1 - dm) / ds), -dm / ds)
                    if (iteration + 1 == max_iterations) or iteration % (max(1, max_iterations / log_num)) == 0:
                        print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')

                    if (iteration + 1) % save_interval == 0 or iteration == 0:
                        self._filter(x_trial)
                        output_den = torch.clamp(x_trial * ds + dm, 0, 1)
                        for j in range(self.num_images):
                            filename = f'rec_{j}.png'
                            torchvision.utils.save_image(output_den[j:j + 1, ...],
                                                         os.path.join(self.exp_dir, filename))
                if self.early_stop and rec_loss < self.loss_thresh:
                    print(f'Early stopping recovery in iteration {iteration}!')
                    break
                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass
        return x_trial.detach(), labels

    def _optimizer(self, variables):
        if self.config['optim'] == 'adam':
            return torch.optim.Adam(variables, lr=self.config['lr'])
        if self.config['optim'] == 'sgd':  # actually gd
            return torch.optim.SGD(variables, lr=SGD_LR, momentum=SGD_MOMENTUM, nesterov=True)
        if self.config['optim'] == 'LBFGS':
            return torch.optim.LBFGS(variables)
        raise ValueError()

    def _filter(self, x_trial):
        """Apply the configured image filter ('none' or 'median') to x_trial in place."""
        if self.config['filter'] == 'none':
            return
        if self.config['filter'] == 'median':
            x_trial.data = MedianPool2d(kernel_size=MEDIAN_FILTER_SIZE, stride=1, padding=1, same=False)(x_trial)
        else:
            raise ValueError()

    def _gradient_closure(self, optimizer, x_trial, input_gradient, label):

        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            loss = self.loss_fn(self.model(x_trial)[0], label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            rec_loss = reconstruction_costs([gradient], input_gradient,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.config['weights'])

            if self.config['total_variation'] > 0:
                rec_loss += self.config['total_variation'] * TV(x_trial)
            if self.config['l2_norm'] > 0:
                rec_loss += self.config['l2_norm'] * torch.norm(x_trial, p=2)
            rec_loss.backward()
            if self.config['signed']:
                x_trial.grad.sign_()
            return rec_loss

        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        if self.config['scoring_choice'] == 'loss':
            self.model.zero_grad()
            x_trial.grad = None
            loss = self.loss_fn(self.model(x_trial)[0], label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            return reconstruction_costs([gradient], input_gradient,
                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                        weights=self.config['weights'])
        elif self.config['scoring_choice'] == 'tv':
            return TV(x_trial)
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(x_trial)
        elif self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            return 0.0
        else:
            raise ValueError()

    def _average_trials(self, x, labels, input_data, stats):
        print(f'Computing a combined result via {self.config["scoring_choice"]} ...')
        if self.config['scoring_choice'] == 'pixelmedian':
            x_optimal, _ = x.median(dim=0, keepdims=False)
        elif self.config['scoring_choice'] == 'pixelmean':
            x_optimal = x.mean(dim=0, keepdims=False)

        self.model.zero_grad()
        if self.reconstruct_label:
            labels = self.model(x_optimal)[0].softmax(dim=1)
        loss = self.loss_fn(self.model(x_optimal)[0], labels)
        gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
        stats['opt'] = reconstruction_costs([gradient], input_data,
                                            cost_fn=self.config['cost_fn'],
                                            indices=self.config['indices'],
                                            weights=self.config['weights'])
        print(f'Optimal result score: {stats["opt"]:2.4f}')
        return x_optimal, stats


def _gradient_indices(indices, input_gradient):
    """Positions in the parameter list whose gradients enter the cost (see --indices)."""
    if isinstance(indices, list):
        return indices
    if indices == 'def':
        return torch.arange(len(input_gradient))
    if indices == 'batch':
        return torch.randperm(len(input_gradient))[:RANDOM_SUBSET_SIZE]
    if indices in TOP_NORM_COUNTS:
        _, top_indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0),
                                    TOP_NORM_COUNTS[indices])
        return top_indices
    if indices in FIRST_COUNTS:
        return torch.arange(0, FIRST_COUNTS[indices])
    if indices in LAST_COUNTS:
        return torch.arange(len(input_gradient))[-LAST_COUNTS[indices]:]
    raise ValueError()


def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal'):
    """Distance between trial gradients and the given gradient, averaged over the trials."""
    indices = _gradient_indices(indices, input_gradient)

    ex = input_gradient[0]
    if weights == 'linear':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
    elif weights == 'exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = input_gradient[0].new_ones(len(input_gradient))

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
        for i in indices:
            if cost_fn == 'l2':
                costs += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum() * weights[i]
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).max() * weights[i]
            elif cost_fn == 'sim':
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
            else:
                raise NotImplementedError('Cost function not implemented.')
        if cost_fn == 'sim':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)
