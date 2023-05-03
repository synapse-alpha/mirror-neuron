import wandb
import os
import time
import tqdm
import pandas as pd
from utils import save_results
from loaders.templates import QueryConfigTemplate

def run_train(model):
    """Run the training loop for a user-specified number of epochs
    """
    config = wandb.config.query
    template = QueryConfigTemplate(**config)
    save_path = template.save_path()
    ignore = template.ignore_attr or {}    
    max_iter = template.method.get('args', {}).get('max_iter',1)

    for i in tqdm.tqdm(range(max_iter)):
        model.train(max_iter=1)

        if i % template.save_interval == 0:
            events = [{k: v for k, v in event.__dict__.items() if k not in ignore} for event in model.history.queue]
            save_results(save_path, events)

def run_forward(model, data):
    """Run the forward pass on the data
    """

    template = QueryConfigTemplate(**wandb.config.query)
    questions, answers = data['question'], data['answer']
    n_sample = len(questions)
    # split data into chunks
    chunks = [questions[i:i+template.chunk_size] for i in range(0, n_sample, template.chunk_size)]

    save_path = template.save_path()
    ignore = template.ignore_attr or {}
    pbar = tqdm.tqdm(chunks, desc='Running reward model', unit='chunk')
    for i, question_chunk in enumerate(pbar):

        # run reward model: this is the main part of the code and should be more configurable
        model.forward(
            roles = ['system', 'user'],
            messages = [ model.config.neuron.base_prompt, question_chunk[0] ], #  chunks are not being used (we only use the first question in the chunk)
            topk = model.config.neuron.training_topk,
            random_sample_uids = True,
            train_gating_model = True,
            timeout = model.config.neuron.training_timeout
        )

        if i * template.chunk_size % template.save_interval == 0:
            events = [{k: v for k, v in event.__dict__.items() if k not in ignore} for event in model.history.queue]
            save_results(save_path, events)

def run_inference(model, data):
    """Run the inference on the data
    """
    pass

def run_query(model, data, **kwargs):
    """Run the query on the data
    """

    template = QueryConfigTemplate(**wandb.config.query)
    print(f'Template: {template}')
    save_path = template.save_path()

    assert len(data) > 0, 'No data to run query on'
    assert model is not None, 'No model provided'

    model.config.neuron.dont_save_events = True

    method_name = template.method.get('name')
    if method_name == 'train':
        run_train(model)
    elif method_name == 'forward':
        run_forward(model, data)
    elif method_name == 'inference':
        run_inference(model, data)
    else:
        raise ValueError(f'Unknown method {method_name!r}')

    # collect every event that was added to internal model queue and save it to a file
    ignore = template.ignore_attr or {}
    events = [{k: v for k, v in event.__dict__.items() if k not in ignore} for event in model.history.queue]

    save_results(save_path, events)
    print(f'+ Saved results to {save_path!r}')

    return pd.DataFrame(events)

