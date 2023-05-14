import datasets
import torch
import random
from dataclasses import dataclass
import queue
import bittensor

from typing import List, Dict, Optional, Callable, Any, Tuple, Union

from base.metagraph import MetagraphMixin

# Here we expose actual dendrite pool
from bittensor._dendrite.text_prompting.dendrite_pool import (
    TextPromptingDendritePool as DendritePool,
)
from sources.neuron import __default_question_prompt__

# TODO: make dendrite pool more interesting by adding custom logic such as a node-specific fail rate, response time or response quality.


@dataclass
class DummyRPCResponse:
    completion: str = None
    alternative: str = None
    question_index: int = None


class DummyDendritePool(MetagraphMixin, torch.nn.Module):
    """Imitates the behaviour of the miner network as defined `via bittensor._dendrite.text_prompting.dendrite_pool`, but returns random data from the dataset instead of querying the network.
    """

    def __init__(
        self,
        data_path: str,
        fail_rate: Union[float, dict] = 0.1,
        fitness: float = 0,
        baseline_fail_rate: float = 0,
        baseline_fitness: float = 0,
        metagraph: 'bittensor.metagraph' = None,
        **kwargs,
    ):
        super(DummyDendritePool, self).__init__()

        self._metagraph = metagraph
        self.data_path = data_path
        self.dataset = datasets.load_dataset(self.data_path)["train"]
        self.questions = [item["question"] for item in self.dataset]
        self.answers = [item["answer"] for item in self.dataset]

        if isinstance(fail_rate, float):
            baseline_fail_rate = fail_rate
            fail_rate: dict = {}
        if not isinstance(fail_rate, dict):
            raise ValueError(
                f"fail_rate must be a float or a dict, got {type(fail_rate)}"
            )

        # let user pass a subset of uids and cast them to int
        self.fail_rate = {int(uid): rate for uid, rate in fail_rate.items()}
        self.baseline_fail_rate = baseline_fail_rate
        self.history: 'queue.Queue' = queue.Queue()

    def apply(self, roles: List[str], messages: List[str], return_call: Any, timeout: int, uid: int) -> 'DummyRPCResponse':

        if random.random() < self.fail_rate.get(int(uid), self.baseline_fail_rate):
            return DummyRPCResponse()

        is_question = messages[-1] == __default_question_prompt__
        # if an answer is requested, return correct answer with some probability
        if not is_question and random.random() < self.fitness.get(int(uid), 0):
            index = self.questions.index(messages[-1])
        else:
            index = random.randint(0, len(self.dataset) - 1)

        question = self.questions[index]
        answer = self.answers[index]
        # select an answer from within an answer list
        answer = answer[random.randint(0, len(answer) - 1)]

        # pad answer with spaces to make it longer than 10 characters
        answer = answer + " " * (10 - len(answer))

        return DummyRPCResponse(
            completion=question if is_question else answer,
            alternative=answer if is_question else question,
            question_index=index,
        )

    def forward(self, roles: List[str], messages: List[str], uids: List[int] = [], return_call: bool = True, timeout: int = 12) -> Union[List['DummyRPCResponse'], None]:
        # BUG? uids should start from 1
        # if uids is None: uids = range( self.metagraph.n.item() )
        if uids is None:
            uids = self.metagraph.uids

        def call_single_uid(uid: int) -> 'DummyRPCResponse':
            return self.apply(
                roles=roles,
                messages=messages,
                return_call=return_call,
                timeout=timeout,
                uid=uid,
            )

        if return_call:
            return [call_single_uid(uid=uid) for uid in uids]
