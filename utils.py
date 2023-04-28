import argparse
import os
import yaml
from yaml.loader import SafeLoader

from config import (
    model_config_template,
    data_config_template,
    query_config_template,
    analysis_config_template
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()


def load_config(path):
    # Open the file and load the file
    with open(path) as f:
        config = yaml.load(f, Loader=SafeLoader)

    return validate_config(config)


def validate_config(config):

    # Load the model
    if config.get('model'):
        config['model'] = _validate_config(config['model'], template=model_config_template, field='model')

    # Load the data
    if config.get('data'):
        config['data'] = _validate_config(config['data'], template=data_config_template, field='data')

    # Run the queries
    if config.get('query'):
        config['query'] = _validate_config(config['query'], template=query_config_template, field='query')

    # Run the analysis
    if config.get('analysis'):
        config['analysis'] = _validate_config(config['analysis'], template=analysis_config_template, field='analysis')

    return config


def _validate_config(config, template, name):
    """
    Check if the config satisfies the template
    """

    assert isinstance(config, dict), "Model must be a dictionary"

    # check if all the required fields are present
    for field, field_info in template.items():
        if field_info['required']:
            assert field in config, f"{name} config must contain {field}"