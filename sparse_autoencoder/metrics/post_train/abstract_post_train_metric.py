"""Abstract post train metric."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from jaxtyping import Float
from torch import Tensor

from sparse_autoencoder.tensor_types import Axis


@dataclass
class PostTrainMetricData:
    """Post Train metric data."""

    input_activations: Float[Tensor, Axis.dims(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)]

    learned_activations: Float[Tensor, Axis.dims(Axis.BATCH, Axis.LEARNT_FEATURE)]

    decoded_activations: Float[Tensor, Axis.dims(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)]

    model: Any



class AbstractPostTrainMetric(ABC):
    """Abstract train metric."""

    @abstractmethod
    def calculate(self, data: PostTrainMetricData) -> dict[str, Any]:
        """Calculate any metrics.

        Args:
            data: Post Train metric data.

        Returns:
            Dictionary of metrics.
        """
