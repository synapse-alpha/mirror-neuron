import os
import time
import pickle
import tqdm
import pandas as pd

def run_query(id, reward_model, data, chunk_size, tokenizer, forward, store, message, entrypoint, **kwargs):
    """
    Reward model and data are not required if loading

    Args:
        id (_type_): _description_
        reward_model (_type_): _description_
        data (_type_): _description_
        chunk_size (_type_): _description_
        tokenizer (_type_): _description_
        forward (_type_): _description_
        store (_type_): _description_
        message (_type_): _description_
        entrypoint (_type_): _description_

    Returns:
        _type_: _description_
    """

    results_path = id
    load_results = True

    if load_results and os.path.exists(results_path):
        # Reuse results from previous run
        return load_results(results_path)

    assert len(data) > 0, 'No data to run query on'
    assert reward_model is not None, 'No reward model provided'

    outputs = []
    n_sample = len(data)
    # split data into chunks
    chunks = [data[i:i+chunk_size] for i in range(0, n_sample, chunk_size)]

    # expected runtime = 0.5 * n_sample
    pbar = tqdm.tqdm(total=len(chunks), desc='Running reward model', unit='chunk')
    for i, chunk in enumerate(pbar):

        t0 = time.time()
        # run reward model: this is the main part of the code and should be more configurable
        out = reward_model.reward(chunk['question'])

        outputs.append({'chunk_size': chunk_size, 'time': time.time() - t0, 'response': out})

        if i * chunk_size % kwargs.get('save_interval', 100) == 0:
            save_results(results_path, outputs)

    save_results(results_path, outputs)

    return pd.DataFrame(outputs)

def save_results(path, outputs):

    if path.endswith('.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(outputs, f)
    elif path.endswith('.csv'):
        df = pd.DataFrame(outputs)
        df.to_csv(path, index=False)
    else:
        raise ValueError(f'Unknown file extension for {path!r}')

def load_results(path):

    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            outputs = pickle.load(f)
            return pd.DataFrame(outputs)
        print(f'Loaded {len(outputs)} results from {path!r}')
    elif path.endswith('.csv'):
        return pd.read_csv(path)
    else:
        raise ValueError(f'Unknown file extension for {path!r}')