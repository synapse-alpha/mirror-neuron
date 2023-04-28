import sys
sys.path.insert(0,'neurons/text/prompting/validators/core/')
from reward import RewardModel
from neuron import neuron

import torch
import os

def load_model(id, path, ckpt, **kwargs):
    """
    Load the model from the path
    """

    config = neuron.config()

    reward_model = RewardModel( model_path = path, device = config.neuron.device)
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

