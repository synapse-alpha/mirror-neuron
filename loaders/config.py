import yaml
from yaml.loader import SafeLoader

from loaders.templates import (
    ModelConfigTemplate,
    DataConfigTemplate,
    QueryConfigTemplate,
    AnalysisConfigTemplate,
)


def load_config(path):
    # Open the file and load the file
    with open(path) as f:
        config = yaml.load(f, Loader=SafeLoader)

    return validate_config(config)


def validate_config(config):

    # Check model config and fill in the defaults
    model_config = config.get("model")
    if model_config:
        model_config = ModelConfigTemplate(**model_config).dict()

    # Check data config and fill in the defaults
    data_config = config.get("data")
    if data_config:
        data_config = DataConfigTemplate(**data_config).dict()

    # Check query config and fill in the defaults
    query_config = config.get("query")
    if query_config:
        query_config = QueryConfigTemplate(**query_config).dict()

    # Check analysis config and fill in the defaults
    analysis_config = config.get("analysis")
    if analysis_config:
        analysis_config = AnalysisConfigTemplate(**analysis_config).dict()

    return config
