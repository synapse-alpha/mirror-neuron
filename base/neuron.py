import torch
import queue
from sources.neuron import neuron

from loguru import logger
import bittensor

from base.reward import ConstantRewardModel
from base.gating import ConstantGatingModel
from base.dendrite_pool import DummyDendritePool


class Neuron(neuron):

    def __init__( self, alpha=0.01, dendrite_pool=None, gating_model=None, reward_model=None, config=None  ):
        self.config = neuron.config() if config is None else config
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
        return f"CustomNeuron(alpha={self.alpha}, dendrite_pool={self.dendrite_pool}, gating_model={self.gating_model}, reward_model={self.reward_model}) using {self.config.neuron.device} device, metagraph={self.metagraph}, wallet={self.wallet}, subtensor={self.subtensor}"

