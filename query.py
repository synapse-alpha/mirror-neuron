import wandb
import os
import time
import tqdm
import pandas as pd
from utils import save_results
from loaders.templates import QueryConfigTemplate


def run_query(model, data, **kwargs):
    """Run the query on the data

    Args:
        model (_type_): _description_
        data (_type_): _description_

    Returns:
        _type_: _description_
    """

    template = QueryConfigTemplate(**wandb.config.query)
    print(f'Template: {template}')
    
    assert len(data) > 0, 'No data to run query on'
    assert model is not None, 'No model provided'
    
    model.config.neuron.dont_save_events = True

    questions, answers = data['question'], data['answer']
    outputs = []
    n_sample = len(questions)
    # split data into chunks
    chunks = [questions[i:i+template.chunk_size] for i in range(0, n_sample, template.chunk_size)]
    
    print(model)
    print(model.gating_model)
    print(model.gating_model.metagraph)
    save_path = template.save_path()

    # expected runtime = 0.5 * n_sample
    pbar = tqdm.tqdm(chunks, desc='Running reward model', unit='chunk')
    for i, question_chunk in enumerate(pbar):

        t0 = time.time()
        # run reward model: this is the main part of the code and should be more configurable
        event = model.forward( 
                    roles = ['system', 'user' ],
                    messages = [ model.config.neuron.base_prompt, question_chunk[0] ], #  chunks are not being used
                    topk = model.config.neuron.training_topk,
                    random_sample_uids = True,
                    train_gating_model = True,
                    timeout = model.config.neuron.training_timeout
                )

        # outputs.append({'time': time.time() - t0, 'event': event})

        if i * template.chunk_size % template.save_interval == 0:
            save_results(save_path, model.history.queue)

    save_results(save_path, model.history.queue)
    print(f'+ Saved results to {save_path!r}')

    return pd.DataFrame(outputs)

