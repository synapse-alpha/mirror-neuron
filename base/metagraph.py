import bittensor
from bittensor._subtensor.chain_data import NeuronInfoLite as NeuronInfoLite

import torch
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Union

from types import SimpleNamespace


class MetagraphMixin:
    @property
    def metagraph(self):
        print(
            f"Calling metagraph getter in {self.__class__.__name__} and self._metagraph is {self._metagraph}"
        )
        assert (
            self._metagraph is not None
        ), "metagraph not set. Please call set_metagraph()"
        return self._metagraph

    @metagraph.setter
    def metagraph(self, metagraph):
        print(
            f"Calling metagraph setter in {self.__class__.__name__} and self._metagraph is {self._metagraph}, it will be set to {metagraph}"
        )
        self._metagraph = metagraph


@dataclass
class DummyAxonInfo(bittensor.axon_info):
    coldkey: str = "000000000000000000000000000000000000000000000000",
    hotkey: str = "000000000000000000000000000000000000000000000000",
    version: int = bittensor.__version_as_int__
    ip: str = "1.2.3.4"  # we want it to look like it is serving
    port: int = 42
    ip_type: int = 4


@dataclass
class DummyMetagraphResponse:
    wallet: str = None
    netuid: int = None
    uids: list = None
    weights: str = None


class BaseMetagraph(torch.nn.Module, ABC):
    def check_config(self):
        pass

    @abstractmethod
    def sync(self):
        pass


class DummyMetagraph(bittensor.metagraph):
    def __init__(
        self,
        netuid: int,
        network: str = "finney",
        lite: bool = True,
        sync: bool = True,
        num_nodes=16,
    ) -> "metagraph":
        super(bittensor.metagraph, self).__init__()
        self.netuid = netuid
        self.network = network
        self.num_nodes = num_nodes

        self.version = torch.nn.Parameter(
            torch.tensor([bittensor.__version_as_int__], dtype=torch.int64),
            requires_grad=False,
        )
        self.n = torch.nn.Parameter(
            torch.tensor([0], dtype=torch.int64), requires_grad=False
        )
        self.block = torch.nn.Parameter(
            torch.tensor([0], dtype=torch.int64), requires_grad=False
        )
        self.uids = torch.nn.Parameter(
            torch.tensor([], dtype=torch.int64), requires_grad=False
        )
        self.axons = []
        if sync:
            self.sync(block=0, lite=lite)
        self.axons = [n.axon_info for n in self.neurons]

        # we add a history queue to keep track the changes in the the metagraph
        self.history = queue.Queue()

    def sync(self, block: Optional[int] = 0, lite: bool = True) -> "metagraph":
        """This method enables the network to sync with the chain, which is an additional level of complexity.
        Since this is a dummy metagraph, we don't need to do anything here as it's not dynamic.
        """
        self.neurons = []

        for i in range(1, self.num_nodes + 1):
            neuron = NeuronInfoLite._null_neuron()
            neuron.uid = i
            neuron.axon_info = DummyAxonInfo(
                hotkey=neuron.hotkey, coldkey=neuron.coldkey
            )
            self.neurons.append(neuron)

        self.lite = lite
        self.n = torch.nn.Parameter(
            torch.tensor(len(self.neurons), dtype=torch.int64), requires_grad=False
        )
        self.version = torch.nn.Parameter(
            torch.tensor([bittensor.__version_as_int__], dtype=torch.int64),
            requires_grad=False,
        )
        self.block = torch.nn.Parameter(
            torch.tensor(block, dtype=torch.int64), requires_grad=False
        )
        self.uids = torch.nn.Parameter(
            torch.tensor([neuron.uid for neuron in self.neurons], dtype=torch.int64),
            requires_grad=False,
        )

    @property
    def hotkeys(self) -> List[str]:
        return [axon.hotkey for axon in self.axons]

    @property
    def coldkeys(self) -> List[str]:
        return [axon.coldkey for axon in self.axons]

    @property
    def addresses(self) -> List[str]:
        return [axon.ip_str() for axon in self.axons]


class Metagraph(bittensor.metagraph):
    def __init__(
        self, netuid: int, network: str = "finney", lite: bool = True, sync: bool = True
    ) -> "metagraph":
        super(Metagraph, self).__init__(netuid, network)

        # we add a history queue to keep track the changes in the the metagraph
        self.history = queue.Queue()
        self.history.put(self.neurons)

    def sync(self, block: Optional[int] = None, lite: bool = True) -> "metagraph":
        # call parent sync
        super(Metagraph, self).sync(block, lite)
        self.history.put(self.neurons)
