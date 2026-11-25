"""Error analysis for ``--analysis``: how well the approximations behind iLRG hold per class."""

import torch

from .utils import AverageMeter, print_util

# Embedding MSEs above this value enter the averages as this value.
EMBEDDING_MSE_CAP = 10

# Metric name -> (label in the log, number format)
METRICS = {
    'approx1_mse': ('Approx1 MSE', '%e'),
    'approx2_mse': ('Approx2 MSE', '%e'),
    'approx3_mse': ('Approx3 MSE', '%e'),
    'approx4_mse': ('Approx4 MSE', '%e'),
    'scale_mse': ('Scale MSE', '%e'),
    'approx_out_mse': ('Approx Out MSE', '%e'),
    'approx_prob_mse': ('Approx Probabilities MSE', '%e'),
    'emb_mse': ('Embedding MSE', '%e'),
    'out_mse': ('Out MSE', '%e'),
    'prob_mse': ('Probabilities MSE', '%e'),
    'emb_sim': ('Embedding Cosine Similarity', '%.3f'),
    'out_sim': ('Out Cosine Similarity', '%.3f'),
    'prob_sim': ('Probabilities Cosine Similarity', '%.3f'),
    'approx1_mre': ('Approx1 MRE', '%.3f'),
    'approx2_mre': ('Approx2 MRE', '%.3f'),
    'approx3_mre': ('Approx3 MRE', '%.3f'),
    'approx4_mre': ('Approx4 MRE', '%.3f'),
    'scale_mre': ('Scale MRE', '%.3f'),
    'approx_out_mre': ('Approx Out MRE', '%.3f'),
    'approx_prob_mre': ('Approx Probabilities MRE', '%.3f'),
    'emb_mre': ('Embedding MRE', '%.3f'),
    'out_mre': ('Out MRE', '%.3f'),
    'prob_mre': ('Probabilities MRE', '%.3f'),
    'soteria_mse': ('Soteria Embedding MSE', '%e'),
    'soteria_sim': ('Soteria Embedding Cosine Similarity', '%.3f'),
}

APPROX_MSES = ('approx1_mse', 'approx2_mse', 'approx3_mse', 'approx4_mse')
APPROX_MRES = ('approx1_mre', 'approx2_mre', 'approx3_mre', 'approx4_mre')
SCALE_MSES = ('scale_mse', 'approx_out_mse', 'approx_prob_mse')
SCALE_MRES = ('scale_mre', 'approx_out_mre', 'approx_prob_mre')
RECOVERY_MSES = ('emb_mse', 'out_mse', 'prob_mse')
RECOVERY_MRES = ('emb_mre', 'out_mre', 'prob_mre')
RECOVERY_SIMILARITIES = ('emb_sim', 'out_sim', 'prob_sim')
SOTERIA = ('soteria_mse', 'soteria_sim')

# Log lines as (metrics, text after the last value). The per-class, per-batch and
# run summary lines differ in order and in which lines end with a space.
CLASS_LINES = [(APPROX_MSES, ''), (SCALE_MSES, ''), (RECOVERY_MSES, ''), (RECOVERY_SIMILARITIES, ''),
               (APPROX_MRES, ''), (SCALE_MRES, ''), (RECOVERY_MRES, ' '), (SOTERIA, ' ')]
BATCH_LINES = [(APPROX_MSES, ''), (SCALE_MSES, ''), (RECOVERY_MSES, ''), (RECOVERY_SIMILARITIES, ' '),
               (APPROX_MRES, ''), (SCALE_MRES, ''), (RECOVERY_MRES, ' '), (SOTERIA, '')]
SUMMARY_LINES = [(APPROX_MSES, ''), (APPROX_MRES, ''), (SCALE_MSES, ''), (SCALE_MRES, ''),
                 (RECOVERY_MSES, ''), (RECOVERY_MRES, ''), (RECOVERY_SIMILARITIES, ''), (SOTERIA, ' ')]
# The run summary spells this label in the singular.
SUMMARY_LABELS = {'prob_sim': 'Probability Cosine Similarity'}


def finite_mean(values):
    """Mean of `values`, counting inf and nan entries (divisions by zero) as zero."""
    values = torch.where(torch.isinf(values), torch.full_like(values, 0), values)
    return torch.mean(torch.where(torch.isnan(values), torch.full_like(values, 0), values))


def mse(estimate, reference):
    return torch.mean((estimate - reference) ** 2)


def mean_relative_error(estimate, reference, scale):
    """Mean of |estimate - reference| / |scale|, ignoring divisions by zero."""
    return finite_mean(torch.abs((estimate - reference) / scale))


def class_errors(model, device, share, rec_emb, rec_prob, w_grad, b_grad, class_stats):
    """Every metric of METRICS for one class that is present in the batch.

    share: fraction of the batch in this class. rec_emb, rec_prob: embedding and probabilities
    restored from the gradients. w_grad, b_grad: this class's row of the (possibly defended)
    fc-layer gradients. class_stats: (embedding, output, probabilities, weight gradient,
    bias gradient), each averaged over the class's samples.
    """
    class_emb, class_out, class_prob, class_wgrad, class_bgrad = class_stats
    fc_out = model.fc(class_emb)
    fc_prob = torch.softmax(fc_out, dim=0)
    out_prob = torch.softmax(class_out, dim=0)
    rec_out = model.fc(rec_emb.to(device))

    class_emb, class_out, class_prob = class_emb.cpu(), class_out.cpu(), class_prob.cpu()
    class_wgrad, class_bgrad = class_wgrad.cpu(), class_bgrad.cpu()
    w_grad, b_grad = w_grad.cpu(), b_grad.cpu()
    gradient_scale = class_wgrad / class_bgrad

    # name: (estimate, reference, scale of the relative error, MSE ignores inf/nan)
    comparisons = {
        'approx1': (class_wgrad, class_bgrad * class_emb, class_wgrad, False),
        'approx2': (w_grad, share * class_wgrad, w_grad, True),
        'approx3': (b_grad, share * class_bgrad, b_grad, True),
        'approx4': (fc_prob.cpu(), class_prob, class_prob, False),
        'scale': (w_grad / b_grad, gradient_scale, gradient_scale, True),
        'approx_out': (fc_out.cpu(), class_out, class_out, False),
        'approx_prob': (out_prob.cpu(), class_prob, class_prob, False),
        'emb': (rec_emb.cpu(), class_emb, class_emb, False),
        'out': (rec_out.cpu(), class_out, class_out, False),
        'prob': (rec_prob.cpu(), class_prob, class_prob, False),
    }
    values = {}
    for name, (estimate, reference, scale, finite_only) in comparisons.items():
        squared_error = (estimate - reference) ** 2
        values[name + '_mse'] = finite_mean(squared_error) if finite_only else torch.mean(squared_error)
        values[name + '_mre'] = mean_relative_error(estimate, reference, scale)
    for name in ('emb', 'out', 'prob'):
        estimate, reference = comparisons[name][:2]
        values[name + '_sim'] = torch.cosine_similarity(estimate, reference, dim=-1)
    values['soteria_mse'] = mse(w_grad, class_emb)
    values['soteria_sim'] = torch.cosine_similarity(w_grad, class_emb, dim=-1)
    return {name: value.item() for name, value in values.items()}


def format_metrics(names, values, label_prefix='', labels=None):
    """'<prefix><label>: <value> | ...' for the given metrics."""
    labels = labels or {}
    parts = []
    for name in names:
        label, number_format = METRICS[name]
        parts.append('%s%s: %s' % (label_prefix, labels.get(name, label), number_format % values[name]))
    return ' | '.join(parts)


def new_meters():
    return {name: AverageMeter() for name in METRICS}


class ErrorAnalysis:
    """Collects per-class errors, prints them and averages them per batch and over the run."""

    def __init__(self, model, device, num_images, ratio, log_file):
        self.model = model
        self.device = device
        self.num_images = num_images
        self.ratio = ratio  # share of extreme values left out at each end of the averages
        self.log_file = log_file
        self.run_meters = new_meters()
        self.batch_meters = new_meters()

    def start_batch(self):
        self.batch_meters = new_meters()

    def add_class(self, class_id, rec_emb, rec_prob, w_grad, b_grad, batch_stats):
        """Compare one class's restored embedding and probabilities with the batch's true values.

        w_grad, b_grad: the class's row of the fc-layer gradients. batch_stats: the per-class
        averages from utils.collect_cls_variables.
        """
        indexes, embeddings, outs, probs, wgrads, bgrads = batch_stats
        if len(indexes[class_id]) == 0:
            # Absent class: only the restored probabilities can be checked (against zero)
            prob_mse = torch.mean(rec_prob ** 2)
            print_util('Class %d | Probabilities MSE: %e' % (class_id, prob_mse.item()), log_file=self.log_file)
            return
        class_stats = (embeddings[class_id], outs[class_id], probs[class_id], wgrads[class_id], bgrads[class_id])
        values = class_errors(self.model, self.device, len(indexes[class_id]) / self.num_images,
                              rec_emb, rec_prob, w_grad, b_grad, class_stats)
        for name, value in values.items():
            if name == 'emb_mse':
                value = value if value < EMBEDDING_MSE_CAP else EMBEDDING_MSE_CAP
            self.batch_meters[name].update(value)
        for names, line_end in CLASS_LINES:
            print_util('Class %d | ' % class_id + format_metrics(names, values) + line_end,
                       log_file=self.log_file)

    def end_batch(self):
        """Trimmed batch averages: add them to the run averages and print them."""
        averages = {}
        for name, meter in self.batch_meters.items():
            meter.filter(self.ratio)
            self.run_meters[name].update(meter.avg)
            averages[name] = meter.avg
        for names, line_end in BATCH_LINES:
            print_util(format_metrics(names, averages, 'Avg ') + line_end, log_file=self.log_file)

    def print_summary(self):
        averages = {name: meter.avg for name, meter in self.run_meters.items()}
        for names, line_end in SUMMARY_LINES:
            print_util(format_metrics(names, averages, 'Mean Avg ', SUMMARY_LABELS) + line_end,
                       log_file=self.log_file)
