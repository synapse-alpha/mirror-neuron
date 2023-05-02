import argparse
import os
import pickle
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument('--entity', type=str, default='monkey-n', help='Wandb entity')
    parser.add_argument('--offline', action='store_true', help='Run offline')
    return parser.parse_args()


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
            events = pickle.load(f)
            print(f'Loaded {len(events)} results from {path!r}')
            return pd.DataFrame(events)
    elif path.endswith('.csv'):
        return pd.read_csv(path)
    else:
        raise ValueError(f'Unknown file extension for {path!r}')