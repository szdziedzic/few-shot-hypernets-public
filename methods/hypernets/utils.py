from typing import Dict

import numpy as np
import torch
from torch import nn
from methods.hypernets.intervalmaml import robust_output


def get_param_dict(net: nn.Module) -> Dict[str, nn.Parameter]:
    """A dict of named parameters of an nn.Module"""
    return {n: p for (n, p) in net.named_parameters()}


def set_from_param_dict(module: nn.Module, param_dict: Dict[str, torch.Tensor]):
    """
    Sets the values of `module` parameters with the values from `param_dict`.

    Works just like:
        nn.Module.load_state_dict()

    with the exception that those parameters are not tunable by default, because
    we set their values to bare tensors instead of nn.Parameter.

    This means that a network with such params cannot be trained directly with an optimizer.
    However, gradients may still flow through those tensors, so it's useful for the use-case of hypernetworks.

    """
    for sdk, v in param_dict.items():
        keys = sdk.split(".")
        param_name = keys[-1]
        m = module
        for k in keys[:-1]:
            try:
                k = int(k)
                m = m[k]
            except:
                m = getattr(m, k)

        param = getattr(m, param_name)
        assert param.shape == v.shape, (sdk, param.shape, v.shape)
        delattr(m, param_name)
        setattr(m, param_name, v)


class SinActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def accuracy_from_scores(scores: torch.Tensor, n_way: int, n_query: int) -> float:
    """Assumes that scores are for examples sorted by class!"""
    best_case_scores = scores[:, 1].squeeze().rename(None)
    s_nq, s_nw = best_case_scores.shape
    assert (s_nq, s_nw) == (n_way * n_query, n_way), ((s_nq, s_nw), (n_query, n_way))
    y_query = np.repeat(range(n_way), n_query)
    worst_case_scores = robust_output(
        scores, torch.from_numpy(y_query).long(), n_way
    )
    topk_scores, topk_labels = best_case_scores.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:, 0] == y_query)
    correct_this = float(top1_correct)
    count_this = len(y_query)
    acc_best_case = correct_this / count_this

    topk_scores, topk_labels = worst_case_scores.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:, 0] == y_query)
    correct_this = float(top1_correct)
    count_this = len(y_query)
    acc_worst_case = correct_this / count_this

    return acc_best_case, acc_worst_case


def kl_diag_gauss_with_standard_gauss(mean, logvar):
    mean_flat = torch.cat([t.view(-1) for t in mean])
    logvar_flat = torch.cat([t.view(-1) for t in logvar])
    var_flat = logvar_flat.exp()

    return -0.5 * torch.sum(1 + logvar_flat - mean_flat.pow(2) - var_flat)


def reparameterize(mu, logvar):
    return mu
