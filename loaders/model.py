import wandb

import base.gating
import base.reward
import base.neuron
import base.metagraph
import base.subtensor
from base.neuron import Neuron
from loaders.templates import ModelConfigTemplate
from inspect import signature

def _load_model_from_module(module, model_type, bt_config, metagraph=None, watch=True, **kwargs):
    """
    Load the model from config
    """
    # convert model_type to class name. model_type is 'gating_model' and class name is 'GatingModel'
    required_class_name = model_type.title().replace('_','')
    choices = [choice for choice in dir(module) if choice.endswith(required_class_name)]

    if kwargs.get('config'):
        config = kwargs['config']
    else:
        config = wandb.config.model.get(model_type, {})

    model_name = config.get('name', None)
    model_args = config.get('args', {})
    # NOTE: Ensure that all models get same device. (Does not carry over from neuron init)
    cls_kwargs = {'config': bt_config, 'metagraph': metagraph, **model_args}
    print(f'\nLooking for {model_name!r} model of type {model_type!r}')

    for cls_name in choices:

        if cls_name == model_name:

            cls = getattr(module, cls_name)
            # get the signature of the class constructor
            valid_kwargs = {k: v for k, v in cls_kwargs.items() if k in signature(cls).parameters}
            print(f'+ Found {cls_name!r} in {model_type!r}. Creating instance with args: {valid_kwargs}')
            model = cls( ** valid_kwargs )
            return model


    raise ValueError(f'Allowed models are {choices}, got {model_name}')


def load_model(bt_config=None, **kwargs):
    """
    Load the model from the path
    """

    template = ModelConfigTemplate(**wandb.config.model)
    print(f'Template: {template}')

    watch = True
    subtensor = _load_model_from_module(base.subtensor, model_type='subtensor', watch=False, bt_config=bt_config, **kwargs)
    metagraph = _load_model_from_module(base.metagraph, model_type='metagraph', watch=False, bt_config=bt_config, **kwargs)
    # NOTE: Do we want to hardcode the base module in here?  What if we want sources.gating, etc?
    dendrite_pool = _load_model_from_module(base.dendrite_pool, model_type='dendrite_pool', watch=watch, bt_config=bt_config, metagraph=metagraph, **kwargs)
    gating_model = _load_model_from_module(base.gating, model_type='gating_model', watch=watch, bt_config=bt_config, metagraph=metagraph, **kwargs)
    reward_model = _load_model_from_module(base.reward, model_type='reward_model', watch=watch, bt_config=bt_config, metagraph=metagraph, **kwargs)
    model = Neuron(
                dendrite_pool=dendrite_pool,
                gating_model=gating_model,
                reward_model=reward_model,
                subtensor=subtensor,
                metagraph=metagraph,
                config=bt_config,
                **kwargs
            )

    print(f'Made model:\n{model}')

    return model

def run_watch_experiment(name):

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np

    # Define a simple sequential model
    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )
    wandb.watch(model, log='all', log_freq=10, log_graph=True)


    # Define a loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Generate a synthetic dataset
    np.random.seed(1234)
    x = np.random.rand(1000, 2)
    y = np.sin(x[:, 0] + x[:, 1])

    # Convert numpy arrays to PyTorch tensors
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    example_ct = 0  # number of examples seen
    # Train the model
    for epoch in range(100):
        # Forward pass
        y_pred = model(x)
        loss = criterion(y_pred.squeeze(), y)

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        example_ct +=  len(x)
        # Print the loss every 10 epochs
        if epoch % 10 == 0:
            wandb.log({f"{name}.epoch": epoch, f"{name}.loss": loss}, step=example_ct)
            print(f'{name} training, Epoch {epoch}, Loss: {loss.item():.4f}')


    # Save the model in the exchangeable ONNX format
    # os.makedirs('./models', exist_ok=True)
    model_path = f'model_{name}.onnx'
    torch.onnx.export(model, x, model_path)
    wandb.save(model_path)
