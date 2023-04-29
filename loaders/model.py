import wandb
import torch
import os
from sources.neuron import neuron

from base.dendrite_pool import DummyDendritePool
from base.gating import DummyGatingModel
from base.reward import DummyRewardModel
from base.neuron import DummyNeuron, CustomNeuron



def load_model(id, path, ckpt, **kwargs):
    """
    Load the model from the path
    """

    wandb.config.get('model')
    config = neuron.config()

    reward_model = DummyRewardModel( model_path = path, device = config.neuron.device)
    # the lines below massive speed up inference (~5x)
    reward_model.eval()
    reward_model.half()
    reward_model.requires_grad_( False )
    # make sure its running on the right device
    reward_model.to(config.neuron.device)

    if ckpt:
        checkpoint_path = os.path.expanduser(config.neuron.reward_path + '/hf_ckpt.pt')
        if not os.path.exists( checkpoint_path ):
            os.makedirs( config.neuron.reward_path, exist_ok = True )
            os.system(
                f"wget -O { checkpoint_path } \
                https://huggingface.co/Dahoas/gptj-rm-static/resolve/main/hf_ckpt.pt"
            )

        ckpt_state = torch.load( checkpoint_path  )
        reward_model.load_state_dict( ckpt_state )
    
    return reward_model

def load_model_2(id, path, ckpt, **kwargs):
    """
    Load the model from the path
    """

    wandb.config.get('model')
    
    model = CustomNeuron( model_path = path, device = config.neuron.device)
    
    return model