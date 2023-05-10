import datasets
import torch

from types import SimpleNamespace
from dataclasses import dataclass

from base.metagraph import MetagraphMixin
import bittensor
from bittensor._dendrite.text_prompting.dendrite_pool import (
    TextPromptingDendritePool as DendritePool,
)

# TODO: make dendrite pool more interesting by adding custom logic such as a node-specific fail rate, response time or response quality.


@dataclass
class DummyRPCResponse:
    completion: str = None
    answer: str = None
    question_index: int = None


class DummyDendritePool(MetagraphMixin, torch.nn.Module):
    """Imitates the behaviour of the miner network as defined `via bittensor._dendrite.text_prompting.dendrite_pool`, but returns random data from the dataset instead of querying the network.
    """

    def __init__(self, data_path, fail_rate=0.1, metagraph=None, **kwargs):
        super(DummyDendritePool, self).__init__()

        self._metagraph = metagraph
        self.data_path = data_path
        self.dataset = datasets.load_dataset(self.data_path)["train"]
        self.fail_rate = fail_rate

    def apply(self, roles, messages, return_call, timeout):

        if torch.rand(1).item() < self.fail_rate:
            return DummyRPCResponse()

        index = torch.randint(0, len(self.dataset) - 1, (1,)).item()
        return DummyRPCResponse(
            completion=self.dataset[index]["question"],
            answer=self.dataset[index]["answer"],
            question_index=index,
        )

    def forward(self, roles, messages, uids=None, return_call=True, timeout=12):

        if uids is None:
            uids = range(self.metagraph.n.item())

        def call_single_uid(uid: int) -> str:
            return self.apply(
                roles=roles, messages=messages, return_call=return_call, timeout=timeout
            )

        return [call_single_uid(uid) for uid in uids]
