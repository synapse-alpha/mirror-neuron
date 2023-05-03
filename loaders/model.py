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


def load_model(bt_config=None, **kwargs):
    """
    Load the model from the path
    """

    template = ModelConfigTemplate(**wandb.config.model)
    print(f'Template: {template}')

    run_watch_experiment()
    dendrite_pool = _load_model_from_module(base.dendrite_pool, 'dendrite_pool', **kwargs)
    wandb.watch(dendrite_pool, log='all')
    gating_model = _load_model_from_module(base.gating, 'gating_model', **kwargs)
    wandb.watch(gating_model, log='all')
    reward_model = _load_model_from_module(base.reward, 'reward_model', **kwargs)
    wandb.watch(reward_model, log='all')
    
    model = Neuron(
                dendrite_pool=dendrite_pool,
                gating_model=gating_model,
                reward_model=reward_model,
                config=bt_config
            )

    print(f'Made model:\n{model}')

    return model

def run_watch_experiment():
        
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
            wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    
