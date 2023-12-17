"""sparsity metric"""
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

class SparsityMetric(AbstractPostTrainMetric):
    """Sparsity Metric.

    """

    # def __init__(self, model):
    #     self.model = model

    @staticmethod
    def sparsity(
        data: PostTrainMetricData,
    ) -> Float[Tensor, Axis.BATCH]:
        """Calculate sparsity.

        Example:

        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data.model.to(device)
        coefficients, _ = data.model(data.input_activations)
        return (coefficients != 0).float().mean(dim=0).sum().item()

    def calculate(self, data: PostTrainMetricData) -> dict[str, Any]:
        """Calculate the fvu for a training batch."""
        sparsity = self.sparsity(data)

        return {
            "sparsity": sparsity,
        }
