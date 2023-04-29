import wandb
import random
import torch
import os
from sources.reward import RewardModel

import bittensor


class DummyRewardModel( torch.nn.Module ):
    # TODO: inherit from RewardModel and just override init
    
    def __init__(self, metagraph, reward_type='question_length'):
        super(DummyRewardModel, self).__init__()
        self.metagraph = metagraph
        self.reward_type = reward_type

    def forward(self, x):
        # each neuron is given a score of 1
        return torch.ones( self.metagraph.n.item() )

    def backward(self, completions, rewards):
        # each neuron is given a score of 1
        return torch.ones( self.metagraph.n.item() )

    def reward(self, completions):
        def reward_fn(samples):
            if self.reward_type == 'question_length':
                rewards = [len(msg) for msg in samples]
            elif self.reward_type == 'longest_word':
                rewards = [len(max(msg.split(), key=len)) for msg in samples]
            elif self.reward_type == 'num_words':
                rewards = [len(msg.split()) for msg in samples]
            return torch.tensor(rewards, dtype=torch.float32).mean()

        rewards = [reward_fn([completion]) for completion in completions]
        return torch.tensor(rewards, dtype=torch.float32)
    
    
class CustomRewardModel( RewardModel ):
    
    def __init__(self, model_path: str, device: str, config: 'bittensor.config' = None, **kwargs):
        super(CustomRewardModel, self).__init__(model_path, device, config)
        
        # just wrecklessly set all the kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
