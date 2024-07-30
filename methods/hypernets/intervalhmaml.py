from copy import deepcopy
from typing import cast
import numpy as np
import torch

from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.autograd import Variable

from backbone import Linear_fw
from methods.hypernets.hypermaml import HyperMAML
from methods.hypernets.utils import accuracy_from_scores, get_param_dict


def radius_transform(radius: Tensor) -> Tensor:
    product = 1
    for dim in list(radius.size()):
        product = dim

    return radius / product


def interval_forward(
    x: Tensor,
    weight: Tensor,
    radius: Tensor,
    bias: Tensor,
    bias_radius: Tensor,
) -> Tensor:
    x = x.rename(None)
    tiler = [1] * (len(x.shape) + 1)
    tiler[1] = 3
    x = x.unsqueeze(1).tile(tiler)
    x = x.refine_names("N", "bounds", "features")
    assert (x.rename(None) >= 0.0).all(), "All input features must be non-negative."

    x_lower, x_middle, x_upper = map(
        lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds")
    )
    assert (
        x_lower <= x_middle
    ).all(), "Lower bound must be less than or equal to middle bound."
    assert (
        x_middle <= x_upper
    ).all(), "Middle bound must be less than or equal to upper bound."

    w_middle = weight
    w_lower = weight - radius_transform(radius)
    w_upper = weight + radius_transform(radius)

    w_lower_pos = w_lower.clamp(min=0)
    w_lower_neg = w_lower.clamp(max=0)
    w_upper_pos = w_upper.clamp(min=0)
    w_upper_neg = w_upper.clamp(max=0)

    # Further splits only needed for numeric stability with asserts
    w_middle_pos = w_middle.clamp(min=0)
    w_middle_neg = w_middle.clamp(max=0)

    lower = x_lower @ w_lower_pos.t() + x_upper @ w_lower_neg.t()
    upper = x_upper @ w_upper_pos.t() + x_lower @ w_upper_neg.t()
    middle = x_middle @ w_middle_pos.t() + x_middle @ w_middle_neg.t()

    if bias is not None:
        bias_scale = 1.0  # TODO: Implement bias scaling
        b_middle = bias + bias_scale * radius_transform(bias_radius)
        b_lower = b_middle - bias_scale * radius_transform(bias_radius)
        b_upper = b_middle + bias_scale * radius_transform(bias_radius)
        lower = lower + b_lower
        upper = upper + b_upper
        middle = middle + b_middle

    assert (
        lower <= middle
    ).all(), "Lower bound must be less than or equal to middle bound."
    assert (
        middle <= upper
    ).all(), "Middle bound must be less than or equal to upper bound."

    return torch.stack([lower, middle, upper], dim=1).refine_names(
        "N", "bounds", "features"
    )


def robust_output(output_lower, output_higher, target, num_classes):
    """Get the robust version of the current output.
    Returns
    -------
    Tensor
        Robust output logits (lower bound for correct class, upper bounds for incorrect classes).
    """
    y_oh = F.one_hot(target, num_classes=num_classes)
    return torch.where(
        y_oh.bool(), output_lower.rename(None), output_higher.rename(None)
    )


class IntervalLinear_fw(
    nn.Linear
):  # used in IntervalHyperMAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(IntervalLinear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None  # Lazy hack to add fast weight link
        self.bias.fast = None
        self.weight.radius = None
        self.bias.radius = None

    def forward(self, x):
        if (
            self.weight.fast is not None
            and self.bias.fast is not None
            and self.weight.radius is not None
            and self.bias.radius is not None
        ):
            out = interval_forward(
                x,
                self.weight.fast,
                self.weight.radius,
                self.bias.fast,
                self.bias.radius,
            )
        else:
            out = super(IntervalLinear_fw, self).forward(x)
        return out


class IntervalHyperNet(nn.Module):
    def __init__(
        self, hn_hidden_size, n_way, embedding_size, feat_dim, out_neurons, params
    ):
        super(IntervalHyperNet, self).__init__()

        self.hn_head_len = params.hn_head_len

        head = [nn.Linear(embedding_size, hn_hidden_size), nn.ReLU()]

        if self.hn_head_len > 2:
            for i in range(self.hn_head_len - 2):
                head.append(nn.Linear(hn_hidden_size, hn_hidden_size))
                head.append(nn.ReLU())

        self.head = nn.Sequential(*head)

        # tails to equate weights with distributions
        tail_mean = [nn.Linear(hn_hidden_size, out_neurons)]
        tail_radius = [nn.Linear(hn_hidden_size, out_neurons)]

        self.tail_mean = nn.Sequential(*tail_mean)
        self.tail_radius = nn.Sequential(*tail_radius)

    def forward(self, x):
        out = self.head(x)
        out_mean = self.tail_mean(out)
        out_radius = self.tail_radius(out)
        return out_mean, out_radius


class IntervalHMAML(HyperMAML):

    def __init__(
        self, model_func, n_way, n_support, n_query, params=None, approx=False
    ):
        super(IntervalHMAML, self).__init__(
            model_func, n_way, n_support, n_query, approx=approx, params=params
        )
        # num of weight set draws for softvoting
        self.eps = params.hm_eps
        self.eps_pump_epochs = params.hm_eps_pump_epochs
        self.eps_pump_value = params.hm_eps_pump_value
        self.radius_eps_warmup_epochs = params.hm_radius_eps_warmup_epochs
        self.worst_case_loss_multiplier = params.hm_worst_case_loss_multiplier

    def _init_classifier(self):
        assert (
            self.hn_tn_hidden_size % self.n_way == 0
        ), f"hn_tn_hidden_size {self.hn_tn_hidden_size} should be the multiple of n_way {self.n_way}"
        layers = []

        for i in range(self.hn_tn_depth):
            in_dim = self.feat_dim if i == 0 else self.hn_tn_hidden_size
            out_dim = (
                self.n_way if i == (self.hn_tn_depth - 1) else self.hn_tn_hidden_size
            )

            if i < self.hn_tn_depth - 1:
                linear = Linear_fw(in_dim, out_dim)
            else:
                linear = IntervalLinear_fw(in_dim, out_dim)

            layers.append(linear)

        self.classifier = nn.Sequential(*layers)

    def _init_hypernet_modules(self, params):
        target_net_param_dict = get_param_dict(self.classifier)

        target_net_param_dict = {
            name.replace(".", "-"): p
            # replace dots with hyphens bc torch doesn't like dots in modules names
            for name, p in target_net_param_dict.items()
        }

        self.target_net_param_shapes = {
            name: p.shape for (name, p) in target_net_param_dict.items()
        }

        self.hypernet_heads = nn.ModuleDict()

        for name, param in target_net_param_dict.items():
            if self.hm_use_class_batch_input and name[-4:] == "bias":
                # notice head_out val when using this strategy
                continue

            bias_size = param.shape[0] // self.n_way

            head_in = self.embedding_size
            head_out = (
                (param.numel() // self.n_way) + bias_size
                if self.hm_use_class_batch_input
                else param.numel()
            )
            # make hypernetwork for target network param
            self.hypernet_heads[name] = IntervalHyperNet(
                self.hn_hidden_size,
                self.n_way,
                head_in,
                self.feat_dim,
                head_out,
                params,
            )

    def get_hn_delta_params(self, support_embeddings):
        if self.hm_detach_before_hyper_net:
            support_embeddings = support_embeddings.detach()

        if self.hm_use_class_batch_input:
            delta_params_list = []

            for name, param_net in self.hypernet_heads.items():

                support_embeddings_resh = support_embeddings.reshape(self.n_way, -1)

                delta_params, params_radius = param_net(support_embeddings_resh)
                bias_neurons_num = self.target_net_param_shapes[name][0] // self.n_way

                if self.hn_adaptation_strategy == "increasing_alpha" and self.alpha < 1:
                    delta_params = delta_params * self.alpha
                    params_radius = params_radius * self.alpha

                weights_delta = (
                    delta_params[:, :-bias_neurons_num]
                    .contiguous()
                    .view(*self.target_net_param_shapes[name])
                )
                bias_delta = delta_params[:, -bias_neurons_num:].flatten()

                weights_radius = (
                    params_radius[:, :-bias_neurons_num]
                    .contiguous()
                    .view(*self.target_net_param_shapes[name])
                )
                bias_radius = params_radius[:, -bias_neurons_num:].flatten()

                delta_params_list.append([weights_delta, weights_radius])
                delta_params_list.append([bias_delta, bias_radius])
            return delta_params_list
        else:
            delta_params_list = []

            for name, param_net in self.hypernet_heads.items():

                flattened_embeddings = support_embeddings.flatten()

                delta_weight, radius = param_net(flattened_embeddings)

                if name in self.target_net_param_shapes.keys():
                    delta_weight = delta_weight.reshape(
                        self.target_net_param_shapes[name]
                    )
                    radius = radius.reshape(self.target_net_param_shapes[name])

                if self.hn_adaptation_strategy == "increasing_alpha" and self.alpha < 1:
                    delta_weight = self.alpha * delta_weight
                    radius = self.alpha * radius

                delta_params_list.append([delta_weight, radius])
            return delta_params_list

    def _update_weight(self, weight, update_weight, radius, train_stage=False):
        if update_weight is None and radius is None:
            return
        weight - update_weight
        if radius is None:  # used in maml warmup
            weight.fast = weight - update_weight
        else:
            weight.radius = torch.abs(radius) + self.eps
            weight.fast = weight - update_weight

    def _update_network_weights(
        self,
        delta_params_list,
        support_embeddings,
        support_data_labels,
        train_stage=False,
    ):
        if self.hm_maml_warmup and not self.single_test:
            p = self._get_p_value()

            # warmup coef p decreases 1 -> 0
            if p > 0.0:
                fast_parameters = []

                clf_fast_parameters = list(self.classifier.parameters())
                for weight in self.classifier.parameters():
                    weight.fast = None
                    weight.radius = None

                self.classifier.zero_grad()
                fast_parameters = fast_parameters + clf_fast_parameters

                for task_step in range(self.task_update_num):
                    scores_lower, scores_middle, scores_upper = self.classifier(
                        support_embeddings
                    )

                    if scores_lower is not None and scores_upper is not None:
                        worst_case_pred = robust_output(
                            scores_lower,
                            scores_upper,
                            support_data_labels,
                            num_classes=self.n_way,
                        )
                        set_loss = self.worst_case_loss_multiplier * self.loss_fn(
                            worst_case_pred, support_data_labels
                        ) + self.loss_fn(scores_middle, support_data_labels)
                    else:
                        set_loss = self.loss_fn(scores_middle, support_data_labels)

                    grad = torch.autograd.grad(
                        set_loss, fast_parameters, create_graph=True, allow_unused=True
                    )  # build full graph support gradient of gradient

                    if self.approx:
                        grad = [
                            g.detach() for g in grad
                        ]  # do not calculate gradient of gradient if using first order approximation

                    if p == 1:
                        # update weights of classifier network by adding gradient
                        for k, weight in enumerate(self.classifier.parameters()):
                            update_value = self.train_lr * grad[k]
                            update_weight, radius = delta_params_list[k]
                            self._update_weight(
                                weight, update_value, radius, train_stage
                            )

                    elif 0.0 < p < 1.0:
                        # update weights of classifier network by adding gradient and output of hypernetwork
                        for k, weight in enumerate(self.classifier.parameters()):
                            update_value = self.train_lr * p * grad[k]
                            update_weight, radius = delta_params_list[k]
                            update_weight = (1 - p) * update_weight + update_value
                            self._update_weight(
                                weight, update_weight, radius, train_stage
                            )
            else:
                for k, weight in enumerate(self.classifier.parameters()):
                    update_weight, radius = delta_params_list[k]
                    self._update_weight(weight, update_weight, radius, train_stage)
        else:
            for k, weight in enumerate(self.classifier.parameters()):
                update_weight, radius = delta_params_list[k]
                self._update_weight(weight, update_weight, radius, train_stage)

    def _get_list_of_delta_params(
        self, maml_warmup_used, support_embeddings, support_data_labels
    ):
        # if not maml_warmup_used:

        if self.enhance_embeddings:
            with torch.no_grad():
                logits = (
                    self.classifier.forward(support_embeddings)
                    .detach()[:, 1]
                    .squeeze()
                    .rename(None)
                )
                logits = F.softmax(logits, dim=1)

            labels = support_data_labels.view(support_embeddings.shape[0], -1)
            support_embeddings = torch.cat((support_embeddings, logits, labels), dim=1)

        for weight in self.parameters():
            weight.fast = None
        for weight in self.classifier.parameters():
            weight.radius = None
        self.zero_grad()

        support_embeddings = self.apply_embeddings_strategy(support_embeddings)

        delta_params = self.get_hn_delta_params(support_embeddings)

        if self.hm_save_delta_params and len(self.delta_list) == 0:
            self.delta_list = [{"delta_params": delta_params}]

        return delta_params

    def forward(self, x):
        out = self.feature.forward(x)

        if self.hm_detach_feature_net:
            out = out.detach()

        scores_lower, scores_middle, scores_upper = self.classifier.forward(out)
        return scores_lower, scores_middle, scores_upper

    def set_forward(self, x, is_feature=False, train_stage=False):
        """1. Get delta params from hypernetwork with support data.
        2. Update target- network weights.
        3. Forward with query data.
        4. Return scores"""

        assert is_feature == False, "MAML do not support fixed feature"

        x = x.cuda()
        x_var = Variable(x)
        support_data = (
            x_var[:, : self.n_support, :, :, :]
            .contiguous()
            .view(self.n_way * self.n_support, *x.size()[2:])
        )  # support data
        query_data = (
            x_var[:, self.n_support :, :, :, :]
            .contiguous()
            .view(self.n_way * self.n_query, *x.size()[2:])
        )  # query data
        support_data_labels = self.get_support_data_labels()

        support_embeddings = self.feature(support_data)

        if self.hm_detach_feature_net:
            support_embeddings = support_embeddings.detach()

        maml_warmup_used = (
            (not self.single_test)
            and self.hm_maml_warmup
            and (self.epoch < self.hm_maml_warmup_epochs)
        )

        delta_params_list = self._get_list_of_delta_params(
            maml_warmup_used, support_embeddings, support_data_labels
        )

        self._update_network_weights(
            delta_params_list, support_embeddings, support_data_labels, train_stage
        )

        if self.hm_set_forward_with_adaptation and not train_stage:
            scores_lower, scores_middle, scores_upper = self.forward(support_data)
            return scores_lower, scores_middle, scores_upper, None
        else:
            if self.hm_support_set_loss and train_stage and not maml_warmup_used:
                query_data = torch.cat((support_data, query_data))

            scores_lower, scores_middle, scores_upper = self.forward(query_data)

            # sum of delta params for regularization
            if self.hm_lambda != 0:
                total_delta_sum = sum(
                    [delta_params.pow(2.0).sum() for delta_params in delta_params_list]
                )

                return scores_lower, scores_middle, scores_upper, total_delta_sum
            else:
                return scores_lower, scores_middle, scores_upper, None

    def set_forward_loss(self, x):
        if (
            self.epoch % self.eps_pump_epochs == 0
            and self.epoch > self.radius_eps_warmup_epochs
        ):
            self.eps = self.eps + self.eps_pump_value

        """Adapt and forward using x. Return scores and total losses"""
        scores_lower, scores_middle, scores_upper, total_delta_sum = self.set_forward(
            x, is_feature=False, train_stage=True
        )

        query_data_labels = Variable(
            torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        ).cuda()
        if self.hm_support_set_loss:
            support_data_labels = torch.from_numpy(
                np.repeat(range(self.n_way), self.n_support)
            ).cuda()
            query_data_labels = torch.cat((support_data_labels, query_data_labels))

        worst_case_pred = None
        loss_worst_case = 0
        loss_best_case = self.loss_fn(scores_middle, query_data_labels)
        if scores_lower is not None and scores_upper is not None:
            worst_case_pred = robust_output(
                scores_lower, scores_upper, query_data_labels, num_classes=self.n_way
            )
            loss_worst_case = self.loss_fn(worst_case_pred, query_data_labels)
        else:
            loss_best_case = self.loss_fn(scores_middle, query_data_labels)

        torch.zeros_like(loss_best_case)

        loss = self.worst_case_loss_multiplier * loss_worst_case + loss_best_case

        if self.hm_lambda != 0:
            loss = loss + self.hm_lambda * total_delta_sum

        y_labels = query_data_labels.cpu().numpy()

        best_case_topk_scores, best_case_topk_labels = scores_middle.data.topk(
            1, 1, True, True
        )
        best_case_topk_ind = best_case_topk_labels.cpu().numpy().flatten()
        best_case_top1_correct = np.sum(best_case_topk_ind == y_labels)
        best_case_task_accuracy = (
            best_case_top1_correct / len(query_data_labels)
        ) * 100

        worst_case_task_accuracy = None
        if scores_lower is not None and scores_upper is not None:
            worst_case_topk_scores, worst_case_topk_labels = worst_case_pred.data.topk(
                1, 1, True, True
            )
            worst_case_topk_ind = worst_case_topk_labels.cpu().numpy().flatten()
            worst_case_top1_correct = np.sum(worst_case_topk_ind == y_labels)
            worst_case_task_accuracy = (
                worst_case_top1_correct / len(query_data_labels)
            ) * 100

        return (
            loss,
            loss_best_case,
            loss_worst_case,
            best_case_task_accuracy,
            worst_case_task_accuracy,
        )

    def set_forward_loss_with_adaptation(self, x):
        """returns loss and accuracy from adapted model (copy)"""
        scores_lower, scores_middle, scores_upper, _ = self.set_forward(
            x, is_feature=False, train_stage=False
        )  # scores from adapted copy
        support_data_labels = Variable(
            torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        ).cuda()

        loss_worst_case = 0
        if scores_lower is not None and scores_upper is not None:
            loss_worst_case = self.loss_fn(
                robust_output(
                    scores_lower,
                    scores_upper,
                    support_data_labels,
                    num_classes=self.n_way,
                ),
                support_data_labels,
            )

        loss_best_case = self.loss_fn(scores_middle, support_data_labels)
        loss = loss_best_case + self.worst_case_loss_multiplier * loss_worst_case

        y_labels = support_data_labels.cpu().numpy()

        best_case_topk_scores, best_case_topk_labels = scores_middle.data.topk(
            1, 1, True, True
        )
        best_case_topk_ind = best_case_topk_labels.cpu().numpy().flatten()
        best_case_top1_correct = np.sum(best_case_topk_ind == y_labels)
        best_case_task_accuracy = (
            best_case_top1_correct / len(support_data_labels)
        ) * 100

        worst_case_task_accuracy = None
        if scores_lower is not None and scores_upper is not None:
            worst_case_topk_scores, worst_case_topk_labels = robust_output(
                scores_lower, scores_upper, support_data_labels, num_classes=self.n_way
            ).data.topk(1, 1, True, True)
            worst_case_topk_ind = worst_case_topk_labels.cpu().numpy().flatten()
            worst_case_top1_correct = np.sum(worst_case_topk_ind == y_labels)
            worst_case_task_accuracy = (
                worst_case_top1_correct / len(support_data_labels)
            ) * 100

        return (
            loss,
            loss_best_case,
            loss_worst_case,
            best_case_task_accuracy,
            worst_case_task_accuracy,
        )

    def train_loop(self, epoch, train_loader, optimizer):  # overwrite parrent function
        print_freq = 10
        avg_loss = 0
        avg_best_case_loss = 0
        avg_worst_case_loss = 0
        task_count = 0
        loss_all = []
        best_case_loss_all = []
        worst_case_loss_all = []
        best_case_acc_all = []
        worst_case_acc_all = []
        optimizer.zero_grad()

        self.delta_list = []

        # train
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML do not support way change"

            (
                loss,
                loss_best_case,
                loss_worst_case,
                best_case_task_accuracy,
                worst_case_task_accuracy,
            ) = self.set_forward_loss(x)
            avg_loss = avg_loss + loss.item()  # .data[0]
            avg_best_case_loss = avg_best_case_loss + loss_best_case.item()
            avg_worst_case_loss = avg_worst_case_loss + loss_worst_case.item()
            loss_all.append(loss)
            best_case_loss_all.append(loss_best_case)
            worst_case_loss_all.append(loss_worst_case)
            best_case_acc_all.append(best_case_task_accuracy)
            worst_case_acc_all.append(worst_case_task_accuracy)

            task_count += 1

            if task_count == self.n_task:  # MAML update several tasks at one time
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()

                optimizer.step()
                task_count = 0
                loss_all = []

            optimizer.zero_grad()
            if i % print_freq == 0:
                print(
                    "Epoch {:d}/{:d} | Batch {:d}/{:d} | Loss {:f} | Best Case Loss {:f} | Worst Case Loss {:f} | Best Case Task Acc {:f} | Worst Case Task Acc {:f}".format(
                        self.epoch,
                        self.stop_epoch,
                        i,
                        len(train_loader),
                        avg_loss / float(i + 1),
                        avg_best_case_loss / float(i + 1),
                        avg_worst_case_loss / float(i + 1),
                        best_case_task_accuracy,
                        worst_case_task_accuracy,
                    )
                )

        best_case_acc_all = np.asarray(best_case_acc_all)
        best_case_acc_mean = np.mean(best_case_acc_all)

        metrics = {"accuracy_best_case/train": best_case_acc_mean}

        worst_case_acc_all = np.asarray(worst_case_acc_all)
        worst_case_acc_mean = np.mean(worst_case_acc_all)

        metrics["accuracy_worst_case/train"] = worst_case_acc_mean

        loss_all = np.asarray(loss_all)
        loss_all_mean = np.mean(loss_all)

        metrics["loss"] = loss_all_mean

        best_case_acc_all = np.asarray(best_case_acc_all)
        best_case_acc_mean = np.mean(best_case_acc_all)

        metrics["loss_best_case"] = best_case_acc_mean

        worst_case_acc_all = np.asarray(worst_case_acc_all)
        worst_case_acc_mean = np.mean(worst_case_acc_all)

        metrics["loss_worst_case"] = worst_case_acc_mean

        metrics["curr_eps"] = self.eps
        metrics["curr_radius"] = self.classifier[-1].radius

        if self.hn_adaptation_strategy == "increasing_alpha":
            metrics["alpha"] = self.alpha

        if self.hm_save_delta_params and len(self.delta_list) > 0:
            delta_params = {"epoch": self.epoch, "delta_list": self.delta_list}
            metrics["delta_params"] = delta_params

        if self.alpha < 1:
            self.alpha += self.hn_alpha_step

        return metrics

    def set_forward_with_adaptation(self, x: torch.Tensor):
        self_copy = deepcopy(self)

        # deepcopy does not copy "fast" parameters so it should be done manually
        for param1, param2 in zip(
            self.feature.parameters(), self_copy.feature.parameters()
        ):
            if hasattr(param1, "fast"):
                if param1.fast is not None:
                    param2.fast = param1.fast.clone()
                else:
                    param2.fast = None

        for param1, param2 in zip(
            self.classifier.parameters(), self_copy.classifier.parameters()
        ):
            if hasattr(param1, "fast"):
                if param1.fast is not None:
                    param2.fast = list(param1.fast)
                else:
                    param2.fast = None
            if hasattr(param1, "radius"):
                if param1.radius is not None:
                    param2.radius = param1.radius.clone()
                else:
                    param2.radius = None

        metrics = {"accuracy/val@-0": self_copy.query_accuracy(x)}

        val_opt_type = (
            torch.optim.Adam if self.hn_val_optim == "adam" else torch.optim.SGD
        )
        val_opt = val_opt_type(self_copy.parameters(), lr=self.hn_val_lr)

        if self.hn_val_epochs > 0:
            for i in range(1, self.hn_val_epochs + 1):
                self_copy.train()
                val_opt.zero_grad()
                (
                    loss,
                    loss_best_case,
                    loss_worst_case,
                    best_case_task_accuracy,
                    worst_case_task_accuracy,
                ) = self_copy.set_forward_loss_with_adaptation(x)
                loss.backward()
                val_opt.step()
                self_copy.eval()
                metrics[f"accuracy/val_support_acc_best_case@-{i}"] = (
                    best_case_task_accuracy
                )
                metrics[f"accuracy/val_support_acc_worst_case@-{i}"] = (
                    worst_case_task_accuracy
                )
                metrics[f"accuracy/val_loss@-{i}"] = loss.item()
                metrics[f"accuracy/val_loss_best_case@-{i}"] = loss_best_case.item()
                metrics[f"accuracy/val_loss_worst_case@-{i}"] = loss_worst_case.item()
                metrics[f"accuracy/val@-{i}"] = self_copy.query_accuracy(x)

        # free CUDA memory by deleting "fast" parameters
        for param in self_copy.parameters():
            param.fast = None
            param.radius = None

        return metrics[f"accuracy/val@-{self.hn_val_epochs}"], metrics

    def query_accuracy(self, x: torch.Tensor) -> float:
        scores_lower, scores_middle, scores_upper, _ = self.set_forward(
            x, train_stage=True
        )
        return 100 * accuracy_from_scores(
            scores_middle, n_way=self.n_way, n_query=self.n_query
        )
