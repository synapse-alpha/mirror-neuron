import wandb
import torch
import datasets
from loaders.templates import DataConfigTemplate


def load_data(**kwargs):
    """
    Load the data from the path

    Returns:
        data: list of dictionaries with keys from schema {question, answer}
    """
    template = DataConfigTemplate(**wandb.config.data)
    print(f"Template: {template}")

    data = datasets.load_dataset(template.path)["train"]

    # apply sampling if not None
    if template.sample is not None:
        n_sample = template.sample["n"]
        print(
            f'Sampling {n_sample} samples from {len(data)} samples using {template.sample["method"]} method'
        )
        if template.sample["method"] == "first":
            return data[:n_sample]
        elif template.sample["method"] == "random":
            n_sample = template.sample["n"]
            seed = template.sample.get("seed", 42)
            generator = torch.Generator().manual_seed(seed)
            return torch.randperm(len(data), generator=generator)[:n_sample]

    print(f"Loaded {len(data)} samples from {template.path}")
    return data
