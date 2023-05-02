import wandb
import torch
import os


import base.gating
import base.reward
import base.neuron
# from base.dendrite_pool import DummyDendritePool
# from base.gating import LongestMessageGatingModel, ConstantGatingModel, RandomGatingModel
# from base.reward import DummyRewardModel, RandomRewardModel, ConstantRewardModel
from base.neuron import DummyNeuron, CustomNeuron
from loaders.templates import ModelConfigTemplate

def _load_model_from_module(module, model_type, **kwargs):
    """
    Load the model from config
    """
    choices = dir(module)
    if kwargs.get('config'):
        config = kwargs['config']
    else:
        config = wandb.config.model.get(model_type, {})

    model_name = config.get('name', None)
    model_args = config.get('args', {})
    print(f'\nLooking for {model_name!r} model of type {model_type!r}')

    # convert model_type to class name. model_type is 'gating_model' and class name is 'GatingModel'
    # required_class_name = model_type.title().replace('_','')
    for cls_name in choices:

        if cls_name == model_name:
            print(f'+ Found {cls_name!r} in {model_type!r}. Creating instance with args: {model_args}')
            cls = getattr(module, cls_name)
            return cls( ** model_args )

    raise ValueError(f'Allowed models are {choices}, got {model_name}')

# def load_dendrite_pool(**kwargs):
#     """
#     Load the dendrite pool from config
#     """
#     allowed_models = {'DummyDendritePool'}
#     dendrite_config = wandb.config.model.get('dendrite_pool', {})
#     print(dendrite_config)
#     model_name = dendrite_config.get('name', None)

#     if model_name == 'DummyDendritePool':
#         dendrite_pool = DummyDendritePool( ** dendrite_config.get('args',{}) )
#     else:
#         raise NotImplementedError(f'Allowed models are {allowed_models}, got {model_name}')
#     return dendrite_pool

# def load_gating_model(**kwargs):
#     """
#     Load the gating model from config
#     """
#     gating_config = wandb.config.model.get('gating_model', {})
#     model_name = gating_config.get('name', None)
#     for cls_name in dir(base.gating):
#         print(cls_name)
#         if cls_name == model_name:
#             cls = getattr(base.gating, cls_name)
#             gating_model = cls( ** gating_config.get('args',{}) )
#             return gating_model

#     raise NotImplementedError(f'Allowed models are {allowed_models}, got {model_name}')


# def load_reward_model(**kwargs):
#     """
#     Load the reward model from config
#     """
#     allowed_models = {'DummyRewardModel', 'ConstantRewardModel', 'RandomRewardModel'}
#     reward_config = wandb.config.model.get('reward_model', {})
#     model_name = reward_config.get('name', None)

#     if model_name == 'DummyRewardModel':
#         reward_model = DummyRewardModel( ** reward_config.get('args',{}) )
#     elif model_name == 'ConstantRewardModel':
#         reward_model = ConstantRewardModel( ** reward_config.get('args',{}) )
#     elif model_name == 'RandomRewardModel':
#         reward_model = RandomRewardModel( ** reward_config.get('args',{}) )
#     else:
#         raise NotImplementedError(f'Allowed models are {allowed_models}, got {model_name}')

#     return reward_model

def load_model(**kwargs):
    """
    Load the model from the path
    """

    template = ModelConfigTemplate(**wandb.config.model)
    print(f'Template: {template}')

    dendrite_pool = _load_model_from_module(base.dendrite_pool, 'dendrite_pool', **kwargs)
    gating_model = _load_model_from_module(base.gating, 'gating_model', **kwargs)
    reward_model = _load_model_from_module(base.reward, 'reward_model', **kwargs)
    model = CustomNeuron(
                alpha=template.alpha,
                dendrite_pool=dendrite_pool,
                gating_model=gating_model,
                reward_model=reward_model,
            )
    
    print(f'Made model:\n{model}')

    return model