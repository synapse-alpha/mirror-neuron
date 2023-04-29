import wandb
import torch
import os
import queue
from sources.neuron import neuron

from loguru import logger
import bittensor

from .reward import RewardModel, CustomRewardModel, DummyRewardModel
from .gating import GatingModel, DummyGatingModel, RandomGatingModel
from .dendrite_pool import DummyDendritePool, DendritePool

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


class CustomNeuron(neuron):

    def __init__( self, alpha=0.01, dendrite_pool=DummyDendritePool, gating_model=DummyGatingModel, reward_model=DummyRewardModel ):
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
        self.dendrite_pool = dendrite_pool( metagraph = self.metagraph, **wandb.config.dendrite_pool )
        # Gating model
        self.gating_model = gating_model( metagraph = self.metagraph, **wandb.config.gating_model )
        # Reward model
        self.reward_model = reward_model( metagraph = self.metagraph, **wandb.config.reward_model )

