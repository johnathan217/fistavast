"""fvu metric"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor

from sparse_autoencoder.metrics.post_train.abstract_post_train_metric import (
    AbstractPostTrainMetric,
    PostTrainMetricData,
)
from sparse_autoencoder.tensor_types import Axis

class FVUMetric(AbstractPostTrainMetric):
    """Fraction Variance Unexplained Metric.

    """

    # def __init__(self, model):
    #     self.model = model

    @staticmethod
    def fvu(
        data: PostTrainMetricData,
    ) -> Float[Tensor, Axis.BATCH]:
        """Calculate fvu.

        Example:

        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        input_activations = data.input_activations.to(device)

        _, x_hat = data.model(input_activations)
        residuals = (input_activations - x_hat).pow(2).mean()
        total = (input_activations - input_activations.mean(dim=0)).pow(2).mean()
        return residuals / total

    def calculate(self, data: PostTrainMetricData) -> dict[str, Any]:
        """Calculate the fvu for a training batch."""
        fvu = self.fvu(data)

        return {
            "fvu": fvu,
        }

