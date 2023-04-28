"""
  Contains the templates for the configuration files.
  We can use these templates to validate the configuration files.

  More complex fields can also be validated by using further custom classes or `pydantic` library.
"""
from dataclasses import dataclass

class BaseConfigTemplate:
    def dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

@dataclass
class ModelConfigTemplate(BaseConfigTemplate):
    id: str
    path: str
    ckpt: str = None
    tokenizer: str = None
    embedding: str = None
    norm: str = None


@dataclass
class DataConfigTemplate(BaseConfigTemplate):
  id: str
  path: str
  schema: dict
  sample: dict = None


@dataclass
class QueryConfigTemplate(BaseConfigTemplate):
  id: str
  requires: list
  entrypoint: str
  chunk_size: int = 1
  message: dict = None
  tokenizer: dict = None
  forward: dict = None
  store: list = None


@dataclass
class AnalysisConfigTemplate(BaseConfigTemplate):
  id: str
  requires: list
  compute: dict = None
  estimators: dict = None
  predict: dict = None
  plot: list = None