import wandb
import torch

from sources.gating import GatingModel
import bittensor



class DummyGatingModel( torch.nn.Module ):
    # TODO: inherit from GatingModel and just override init

    def __init__(self, metagraph, type='longest'):
        super(DummyGatingModel, self).__init__()
        self.metagraph = metagraph

    def forward(self, x):
        # each neuron is given a score of 1
        return torch.ones( self.metagraph.n.item() )

    def backward(self, scores, rewards):

        # each neuron is given a score of 1
        return torch.ones( self.metagraph.n.item() )


class RandomGatingModel( torch.nn.Module ):
    # TODO: inherit from GatingModel and just override init

    def __init__(self, metagraph):
        super(RandomGatingModel, self).__init__()
        self.metagraph = metagraph

    def forward(self, x):
        # each neuron is given a random score
        return torch.random( self.metagraph.n.item() )

    def backward(self, scores, rewards):

        # each neuron is given a random score
        return torch.random( self.metagraph.n.item() )