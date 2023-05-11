import bittensor
import torch
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass

from types import SimpleNamespace


@dataclass
class DummySubstrateResponse:
    wallet: str = None
    netuid: int = None
    uids: list = None
    weights: str = None


@dataclass
class BaseSubtensor(torch.nn.Module, ABC):

    network: str
    block: int
    epoch_length: int
    delegated: dict
    metagraph: bittensor.metagraph = None

    def check_config(self):
        pass

    def serve_axon(self, netuid, axon):
        pass

    def _check_delegated(self, delegated):
        """Ensure that delegated has expected behavior
        self.my_nominators = { nomin[0]: nomin[1] for nomin in self.subtensor.get_delegated( self.wallet.coldkeypub.ss58_address )[0][0].nominators }
        """
        if delegated is None:
            delegated = [[SimpleNamespace(nominators=())]]
        else:
            try:
                nominators = {
                    nomin[0]: nomin[1] for nomin in delegated[0][0].nominators
                }
            except Exception as e:
                raise ValueError(
                    "delegated must be a list of lists of SimpleNamespace(nominators=())"
                )

        return delegated

    def validator_epoch_length(self, netuid):
        return self.epoch_length

    @abstractmethod
    def get_delegated(self, ss58_address):
        pass

    @abstractmethod
    def set_weights(self, wallet, netuid, uids, weights, wait_for_finalization=True):
        pass

    @abstractmethod
    def neurons(self, netuid):
        pass


class DummySubtensor(BaseSubtensor):
    def __init__(
        self,
        network="mirror",
        block=1,
        epoch_length=100,
        delegated=None,
        metagraph=None,
        config=None,
    ):
        self.network = network
        self.block = block
        self.epoch_length = epoch_length
        self.delegated = self._check_delegated(delegated)
        self.metagraph = metagraph
        self.config = config
        self.history = queue.Queue()

    def get_delegated(self, ss58_address):
        return self.delegated

    def record_event(self, event: SimpleNamespace):
        self.history.put(event)

    def set_weights(self, wallet, netuid, uids, weights, wait_for_finalization=True):
        event = DummySubstrateResponse(
            wallet=wallet, netuid=netuid, uids=uids, weights=weights
        )
        self.history.put(event)

    def neurons(self, netuid):
        return self.metagraph.neurons


class Subtensor(bittensor.subtensor):
    def __init__(self, config):
        super(Subtensor, self).__init__(config)
