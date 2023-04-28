import wandb
import torch
import os
import queue
import sys
sys.path.insert(0,'neurons/text/prompting/validators/core/')
from neuron import neuron

from loguru import logger
import bittensor as bt

from .reward import RewardModel, CustomRewardModel, DummyRewardModel
from .gating import GatingModel, DummyGatingModel, RandomGatingModel
from .dendrite_pool import DummyDendritePool, DendritePool

class DummyNeuron(neuron):

    def check_config( cls, config ):
        # i'm only overriding this method to prevent the checkpoint from being downloaded. everything else is the same.
        
        full_path = os.path.expanduser('{}/{}/{}/netuid{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.netuid, config.neuron.name ))
        config.neuron.full_path = os.path.expanduser( full_path )
        config.neuron.reward_path = os.path.expanduser( config.neuron.reward_path )

        # Add custom event logger for the events.
        logger.level("EVENTS", no=38, icon="📝")
        logger.add(
            config.neuron.full_path + "/" + "completions.log",
            rotation=config.neuron.events_retention_size, serialize=True, enqueue=True, backtrace=False, diagnose=False, level="EVENTS",
            format = "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message} | {extra[prompt]} {extra[completion]} {extra[uids]} {extra[all_uids]} {extra[rewards]} {extra[scores]} {extra[all_completions]} {extra[block]}"
        )

    def __init__( self, alpha=0.01 ):
        self.config = neuron.config()
        self.check_config( self.config )
        self.alpha = alpha
        bt.logging( config = self.config, logging_dir = self.config.neuron.full_path )

        self.subtensor = bt.subtensor ( config = self.config )
        self.device = torch.device( self.config.neuron.device )
        self.wallet = bt.wallet ( config = self.config )
        self.metagraph = bt.metagraph( netuid = self.config.netuid, network = self.subtensor.network )
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
        bt.logging( config = self.config, logging_dir = self.config.neuron.full_path )

        self.subtensor = bt.subtensor ( config = self.config )
        self.device = torch.device( self.config.neuron.device )
        self.wallet = bt.wallet ( config = self.config )
        self.metagraph = bt.metagraph( netuid = self.config.netuid, network = self.subtensor.network )
        self.wallet.create_if_non_existent()

        # History of forward events.
        self.history = queue.Queue( maxsize = self.config.neuron.max_history )
        # Dendrite pool
        self.dendrite_pool = dendrite_pool( metagraph = self.metagraph, **wandb.config.dendrite_pool )
        # Gating model
        self.gating_model = gating_model( metagraph = self.metagraph, **wandb.config.gating_model )
        # Reward model
        self.reward_model = reward_model( metagraph = self.metagraph, **wandb.config.reward_model )

