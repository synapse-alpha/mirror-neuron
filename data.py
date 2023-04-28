import datasets
import numpy as np

def load_data(id, path, schema, sample, **kwargs):
    """
    Load the data from the path

    Returns:
        data: list of dictionaries with keys from schema {question, answer}
    """
    data = datasets.load_dataset(path)

    data = data['train']

    # apply sampling if not None
    if sample is not None:
        n_sample = sample['n']
        if sample['method'] == 'first':
            indices = np.arange(n_sample)
        elif sample['method'] == 'random':
            n_sample = sample['n']
            seed = sample.get('seed', 42)
            indices = np.random.RandomState(seed=seed).choice(len(data['train']), n_sample, replace=False)

        data = [data[i] for i in indices]

    return data