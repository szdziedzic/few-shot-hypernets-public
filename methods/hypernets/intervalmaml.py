import math
import os
from abc import ABC
from enum import Enum
from typing import cast, List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_decision_regions

RADIUS_MIN = 0.0


class IntervalModuleWithWeights(nn.Module, ABC):
    def __init__(self):
        super().__init__()


class IntervalLinear(IntervalModuleWithWeights):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_radius: float,
        bias: bool,
        initial_radius: float,
        normalize_shift: bool,
        normalize_scale: bool,
        scale_init: float = -5.0,
        initial_eps=0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_radius = max_radius
        self.normalize_shift = normalize_shift
        self.normalize_scale = normalize_scale
        self.scale_init = scale_init
        self.initial_radius = initial_radius
        self.eps = initial_eps

        assert self.max_radius > 0

        self.weight = Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty(out_features), requires_grad=True)
        else:
            self.bias = None
        self._radius = torch.empty((out_features, in_features)).cuda()
        # self._shift = Parameter(
        #     torch.empty((out_features, in_features)), requires_grad=False
        # )
        # self._scale = Parameter(
        #     torch.empty((out_features, in_features)), requires_grad=False
        # )

        # TODO test and fix so that it still works with bias=False
        # if bias:
        #     self.bias = Parameter(torch.empty(out_features), requires_grad=True)
        # else:
        #     self.bias = None
        # self.mode: Mode = Mode.VANILLA
        self.reset_parameters()

    def radius_transform(self, params: Tensor):
        # self._radius clamp
        # gamble softmax
        return params

    @property
    def radius(self) -> Tensor:
        return self.radius_transform(self._radius)

    @property
    def bias_radius(self) -> Tensor:
        return self.radius_transform(self._bias_radius)

    @property
    def shift(self) -> Tensor:
        """Contracted interval middle shift (-1, 1)."""
        if self.normalize_shift:
            eps = torch.tensor(1e-8).to(self._shift.device)
            return (self._shift / torch.max(self.radius, eps)).tanh()
        else:
            return self._shift.tanh()

    @property
    def bias_shift(self) -> Tensor:
        """Contracted interval middle shift (-1, 1)."""
        if self.normalize_shift:
            eps = torch.tensor(1e-8).to(self._bias_shift.device)
            return (self._bias_shift / torch.max(self.bias_radius, eps)).tanh()
        else:
            return self._bias_shift.tanh()

    @property
    def scale(self) -> Tensor:
        """Contracted interval scale (0, 1)."""
        if self.normalize_scale:
            eps = torch.tensor(1e-8).to(self._scale.device)
            scale = (self._scale / torch.max(self.radius, eps)).sigmoid()
        else:
            scale = self._scale.sigmoid()
        return scale * (1.0 - torch.abs(self.shift))

    @property
    def bias_scale(self) -> Tensor:
        """Contracted interval scale (0, 1)."""
        if self.normalize_scale:
            eps = torch.tensor(1e-8).to(self._bias_scale.device)
            scale = (self._bias_scale / torch.max(self.radius, eps)).sigmoid()
        else:
            scale = self._bias_scale.sigmoid()
        return scale * (1.0 - torch.abs(self.bias_shift))

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore
            self._radius.zero_()
            # self._shift.zero_()
            # self._scale.fill_(self.scale_init)
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

                self.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        x = x.refine_names("N", "bounds", "features")  # type: ignore
        assert (x.rename(None) >= 0.0).all(), "All input features must be non-negative."  # type: ignore

        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        assert (
            x_lower <= x_middle
        ).all(), "Lower bound must be less than or equal to middle bound."
        assert (
            x_middle <= x_upper
        ).all(), "Middle bound must be less than or equal to upper bound."

        w_middle: Tensor = self.weight
        w_lower = self.weight - self.radius
        w_upper = self.weight + self.radius

        # print(f"DDDD ${self.radius}")

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

        if self.bias is not None:
            b_middle = self.bias  # + self.bias_radius
            b_lower = b_middle  # - self.bias_radius
            b_upper = b_middle  # + self.bias_radius
            lower = lower + b_lower
            upper = upper + b_upper
            middle = middle + b_middle

        assert (
            lower <= middle
        ).all(), "Lower bound must be less than or equal to middle bound."
        assert (
            middle <= upper
        ).all(), "Middle bound must be less than or equal to upper bound."

        return torch.stack([lower, middle, upper], dim=1).refine_names("N", "bounds", "features")  # type: ignore


class DeIntervaler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.refine_names("N", "bounds", ...)
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        return x_middle


class ReIntervaler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.rename(None)
        tiler = [1] * (len(x.shape) + 1)
        tiler[1] = 3
        x = x.unsqueeze(1).tile(tiler)
        return x


class IntervalModel(nn.Module):
    def __init__(self, max_radius: float):
        super().__init__()
        self._max_radius = max_radius

    def interval_children(self) -> List[IntervalModuleWithWeights]:
        return [m for m in self.modules() if isinstance(m, IntervalModuleWithWeights)]

    def named_interval_children(self) -> List[Tuple[str, IntervalModuleWithWeights]]:
        return [
            (n, m)
            for n, m in self.named_modules()
            if isinstance(m, IntervalModuleWithWeights)
        ]

    @property
    def max_radius(self):
        return self._max_radius

    @max_radius.setter
    def max_radius(self, value: float) -> None:
        self._max_radius = value
        for m in self.interval_children():
            m.max_radius = value

    def clamp_radii(self) -> None:
        for m in self.interval_children():
            m.clamp_radii()

    def radius_transform(self, params: Tensor) -> Tensor:
        for m in self.interval_children():
            return m.radius_transform(params)
        raise ValueError("No IntervalNet modules found in model.")


class IntervalMLP(IntervalModel):
    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        output_classes: int,
        max_radius: float,
        initial_radius: float,
        bias: bool,
        normalize_shift: bool,
        normalize_scale: bool,
        scale_init: float,
    ):
        super().__init__(max_radius=max_radius)

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_classes = output_classes
        self.normalize_shift = normalize_shift
        self.normalize_scale = normalize_scale
        self.interval_linear = IntervalLinear(
            self.input_size,
            self.output_classes,
            max_radius=max_radius,
            initial_radius=initial_radius,
            bias=bias,
            normalize_shift=normalize_shift,
            normalize_scale=normalize_scale,
            scale_init=scale_init,
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:  # type: ignore
        # x = x.refine_names("N", "C", "H", "W")  # type: ignore  # expected input shape
        # x = x.rename(None)  # type: ignore  # drop names for unsupported operations
        x = x.flatten(1)  # (N, features)
        x = x.unflatten(1, (1, -1))  # type: ignore  # (N, bounds, features)
        x = x.tile((1, 3, 1))

        x = x.refine_names("N", "bounds", "features")  # type: ignore

        return F.relu(self.interval_linear(x))

    @property
    def device(self):
        return self.interval_linear.weight.device


def robust_output(last_output, target, num_classes, device=torch.device("cuda")):
    """Get the robust version of the current output.
    Returns
    -------
    Tensor
        Robust output logits (lower bound for correct class, upper bounds for incorrect classes).
    """
    output_lower, _, output_higher = last_output.unbind("bounds")
    y_oh = F.one_hot(target, num_classes=num_classes)  # type: ignore
    y_oh = y_oh.to(device)
    return torch.where(
        y_oh.bool(), output_lower.rename(None), output_higher.rename(None)
    )
