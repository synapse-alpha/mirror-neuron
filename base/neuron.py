import wandb
import torch
import os
import queue
from sources.neuron import neuron

from loguru import logger
import bittensor

from base.reward import DummyRewardModel, ConstantRewardModel, RandomRewardModel
from base.gating import LongestMessageGatingModel, ConstantGatingModel, RandomGatingModel
from base.dendrite_pool import DummyDendritePool, DendritePool

class DummyNeuron(neuron):

    def __init__( self, alpha=0.01 ):
        self.config = neuron.config()
        self.check_config( self.config )
        self.alpha = alpha
        bittensor.logging( config = self.config, logging_dir = self.config.neuron.full_path )

        self.subtensor = bittensor.subtensor ( config = self.config )
        self.device = torch.device( self.config.neuron.device )
        self.wallet = bittensor.wallet ( config = self.config )
        self.metagraph = bittensor.metagraph( netuid = self.config.netuid, network = self.subtensor.network )
        self.wallet.create_if_non_existent()

        # History of forward events.
        self.history = queue.Queue( maxsize = self.config.neuron.max_history )
        # Dendrite pool
        self.dendrite_pool = DummyDendritePool( metagraph = self.metagraph )
        # Gating model
        self.gating_model = DummyGatingModel( metagraph = self.metagraph )
        # Reward model
        self.reward_model = DummyRewardModel( metagraph = self.metagraph, reward_type='question_length' )


class Neuron(neuron):

    def __init__( self, alpha=0.01, dendrite_pool=None, gating_model=None, reward_model=None ):
        self.config = neuron.config()
        self.check_config( self.config )
        self.alpha = alpha # for weight updating
        bittensor.logging( config = self.config, logging_dir = self.config.neuron.full_path )

        self.subtensor = bittensor.subtensor ( config = self.config )
        self.device = torch.device( self.config.neuron.device )
        self.wallet = bittensor.wallet ( config = self.config )
        self.metagraph = bittensor.metagraph( netuid = self.config.netuid, network = self.subtensor.network )
        self.wallet.create_if_non_existent()

        # History of forward events.
        self.history = queue.Queue( maxsize = self.config.neuron.max_history )

        # Dendrite pool
        if dendrite_pool is None:
            dendrite_pool = DummyDendritePool(metagraph = self.metagraph)
        else:
            dendrite_pool.metagraph = self.metagraph
        self.dendrite_pool = dendrite_pool

        # Gating model
        if gating_model is None:
            gating_model = ConstantGatingModel(metagraph = self.metagraph)
        else:
            gating_model.metagraph = self.metagraph
        self.gating_model = gating_model

        # Reward model
        if reward_model is None:
            reward_model = ConstantRewardModel(metagraph = self.metagraph)
        else:
            reward_model.metagraph = self.metagraph
        self.reward_model = reward_model

    def __repr__(self):
        return "CustomNeuron(alpha={}, dendrite_pool={}, gating_model={}, reward_model={})".format(self.alpha, self.dendrite_pool, self.gating_model, self.reward_model)

