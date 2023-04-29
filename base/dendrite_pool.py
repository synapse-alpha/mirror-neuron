import wandb
import random
import datasets
import torch

from types import SimpleNamespace
from dataclasses import dataclass

import bittensor
from bittensor._dendrite.text_prompting.dendrite_pool import TextPromptingDendritePool as DendritePool

@dataclass
class DummyRPCResponse:
    completion: str = None
    answer: str = None
    question_index: int = None
    
    
class DummyDendritePool( torch.nn.Module ):
    
    def __init__(self, metagraph, fail_rate=0.1):
        super(DummyDendritePool, self).__init__()
        self.metagraph = metagraph
        self.data_path = wandb.config.data_path
        self.dataset = datasets.load_dataset(self.data_path)['train']
        self.fail_rate = fail_rate

    def query(self, roles, messages, return_call, timeout):
        
        if random.random() < self.fail_rate:
            return DummyRPCResponse()
        
        index = random.randint(0, len(self.dataset)-1)
        return DummyRPCResponse(
            completion = self.dataset[index]['question'],
            answer = self.dataset[index]['answer'],
            question_index = index,
        )

    def forward(self, roles, messages, uids=None, return_call=True, timeout=12):

        if uids is None: uids = range( self.metagraph.n.item() )

        def call_single_uid( uid: int ) -> str:
            return self.query(
                roles = roles,
                messages = messages,
                return_call = return_call,
                timeout = timeout
                )

        return [ call_single_uid( uid ) for uid in uids ]

