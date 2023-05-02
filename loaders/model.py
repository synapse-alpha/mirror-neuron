import wandb

import base.gating
import base.reward
import base.neuron
from base.neuron import Neuron
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


def load_model(**kwargs):
    """
    Load the model from the path
    """

    template = ModelConfigTemplate(**wandb.config.model)
    print(f'Template: {template}')

    dendrite_pool = _load_model_from_module(base.dendrite_pool, 'dendrite_pool', **kwargs)
    gating_model = _load_model_from_module(base.gating, 'gating_model', **kwargs)
    reward_model = _load_model_from_module(base.reward, 'reward_model', **kwargs)
    model = Neuron(
                alpha=template.alpha,
                dendrite_pool=dendrite_pool,
                gating_model=gating_model,
                reward_model=reward_model,
            )

    print(f'Made model:\n{model}')

    return model