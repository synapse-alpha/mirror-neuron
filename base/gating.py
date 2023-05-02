import torch

from base.values import ConstantValue, RandomValue, FrozenRandomValue
from base.metagraph import MetagraphMixin
from abc import ABC, abstractmethod

# from sources.gating import GatingModel
# TODO: inherit from GatingModel and just override init



class BaseGatingModel( torch.nn.Module ):

    def __init__(self, metagraph):
        super(BaseGatingModel, self).__init__()
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

class LongestMessageGatingModel( BaseGatingModel ):

    def __init__(self, metagraph=None):
        super(LongestMessageGatingModel, self).__init__( )
        self._metagraph = metagraph

    def forward(self, x, n):
        # return ones for the longest n messages and zeros for the rest
        threshold = torch.topk(x, n, largest=False).values[-1]
        return torch.ones( x.shape ) * (x >= threshold).float()

    def backward(self, scores, rewards):

        return torch.ones( self.metagraph.n.item() )


class RandomGatingModel( BaseGatingModel ):

    def __init__(self, frozen=False, seed=0, distribution='uniform', p0=1, p1=0, metagraph=None):
        super(RandomGatingModel, self).__init__( metagraph=metagraph )
        
        value_type = FrozenRandomValue if frozen else RandomValue
        self.value = value_type(seed=seed, distribution=distribution, p0=p0, p1=p1)

    def forward(self, x):
        # each neuron is given a random score
        return self.value(x, self.metagraph.n.item())

    def backward(self, scores, rewards):
        # each neuron is given a random score
        return self.value(torch.zeros( self.metagraph.n.item() ), self.metagraph.n.item())


class ConstantGatingModel( BaseGatingModel ):

    def __init__(self, value=1, metagraph=None):
        super(ConstantGatingModel, self).__init__( metagraph=metagraph )
        self.value = ConstantValue(value)

    def forward(self, x):
        # each neuron is given a constant score
        return self.value(x, self.metagraph.n.item())

    def backward(self, scores, rewards):

        # each neuron is given a random score
        return torch.random( self.metagraph.n.item() )


class MaskedGatingModel( BaseGatingModel ):
    
    def __init__(self, mask=10, metagraph=None):
        super(ConstantGatingModel, self).__init__( metagraph=metagraph )
        raise NotImplementedError(f'Not implemented yet.')