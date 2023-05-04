import bittensor
import torch

from base.values import ConstantValue, RandomValue
from abc import ABC, abstractmethod

# expose raw RewardModel for use in other modules
from sources.reward import RewardModel

# include query failure as an additional behavior
# TODO: inherit from RewardModel and just override init

class BaseRewardModel( torch.nn.Module, ABC ):

    def __init__(self, metagraph, **kwargs):
        super(BaseRewardModel, self).__init__()
        self._metagraph = metagraph
    
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, completions, rewards):
        pass
    
    @abstractmethod
    def reward(self, completions):
        pass


class DummyRewardModel( BaseRewardModel ):
    
    def __init__(self, reward_type='question_length', forward_value=ConstantValue(value=1), backward_value=ConstantValue(value=1), metagraph=None, **kwargs):
        super(DummyRewardModel, self).__init__( metagraph=metagraph )
        self.reward_type = reward_type
        self.forward_value = forward_value
        self.backward_value = backward_value

    def forward(self, x):
        # each neuron is given a score of 1
        return self.forward_type(x, self.metagraph.n.item())

    def backward(self, completions, rewards):
        # each neuron is given a score of 1
        return self.backward_value(completions, rewards, n=self.metagraph.n.item())

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

class ConstantRewardModel( BaseRewardModel ):
    
    def __init__(self, forward_value=1, backward_value=0, metagraph=None, **kwargs):
        super(ConstantRewardModel, self).__init__( metagraph=metagraph )
        self.forward_value = ConstantValue(value=forward_value)
        self.backward_value = ConstantValue(value=backward_value)

    def forward(self, x):
        # each neuron is given a constant score
        return self.forward_value(x, n=1)

    def backward(self, completions, rewards, n=1):
        # each neuron is given a constant score
        return self.backward_value(completions, rewards, n=self.metagraph.n.item())

    def reward(self, completions):
        
        scores = [self.forward([completion]) for completion in completions]
        return torch.tensor(scores, dtype=torch.float32)

class RandomRewardModel( BaseRewardModel ):
    
    def __init__(self, seed=0, distribution='uniform', p0=1, p1=0, metagraph=None, **kwargs):
        super(RandomRewardModel, self).__init__( metagraph=metagraph )
        self.forward_value = RandomValue(seed=seed, distribution=distribution, p0=p0, p1=p1)
        self.backward_value = RandomValue(seed=seed, distribution=distribution, p0=kwargs.get('backward_p0',p0), p1=kwargs.get('backward_p1',p1))

    def forward(self, x):
        # each neuron is given a constant score
        return self.forward_value(x, n=1)

    def backward(self, completions, rewards):
        # each neuron is given a constant score
        return self.backward_value(completions, rewards, n=self.metagraph.n.item())

    def reward(self, completions):

        return torch.tensor([self.forward([completion]) for completion in completions], dtype=torch.float32)
    
class CustomRewardModel( RewardModel ):
    
    def __init__(self, model_path: str, device: str, config: 'bittensor.config' = None, **kwargs):
        super(CustomRewardModel, self).__init__(model_path, device, config)
        
        # just wrecklessly set all the kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
