import logging
from typing import Iterable, Tuple

import torch.nn as nn

from ..layer import AdapterLayer
from ..model_mixin import InvertibleAdaptersMixin, ModelAdaptersMixin


logger = logging.getLogger(__name__)


# For backwards compatibility, BertSelfOutput inherits directly from AdapterLayer
class BertSelfOutputAdaptersMixin(AdapterLayer):
    """Adds adapters to the BertSelfOutput module."""

    def __init__(self,args):
        super().__init__("mh_adapter", None,args=args)
        self.args = args


# For backwards compatibility, BertOutput inherits directly from AdapterLayer
class BertOutputAdaptersMixin(AdapterLayer):
    """Adds adapters to the BertOutput module."""

    def __init__(self,args):
        super().__init__("output_adapter", None,args=args)
        self.args = args


class BertModelAdaptersMixin(InvertibleAdaptersMixin, ModelAdaptersMixin):
    """Adds adapters to the BertModel module."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.encoder.layer):
            yield i, layer
