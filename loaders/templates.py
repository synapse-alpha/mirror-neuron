"""
  Contains the templates for the configuration files.
  We can use these templates to validate the configuration files.

  More complex fields can also be validated by using further custom classes or `pydantic` library.
  
  @sarthakbatragatech: I think
  these templates should actually be used to create the config files (good idea copilot)
  but i was actually trying to say that the validation should come from trying to instantiate the models themselves (although we don't want to download or load the models themselves at that point)
"""
from dataclasses import dataclass

class BaseConfigTemplate:
    def dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save_path(self):
      return self.id+'.pkl'

@dataclass
class BaseModelConfigTemplate(BaseConfigTemplate):
    id: str
    path: str
    args: dict = None
    ckpt: str = None
    tokenizer: str = None
    embedding: str = None
    norm: str = None

@dataclass
class ModelConfigTemplate(BaseConfigTemplate):
    id: str
    dendrite_pool: BaseModelConfigTemplate = None
    gating_model: BaseModelConfigTemplate = None
    reward_model: BaseModelConfigTemplate = None
    alpha: float = 0.01
    


@dataclass
class DataConfigTemplate(BaseConfigTemplate):
  id: str
  path: str
  schema: dict
  sample: dict = None


@dataclass
class QueryConfigTemplate(BaseConfigTemplate):
  id: str
  chunk_size: int = 1
  save_interval: int = 100
  message: dict = None
  ignore_attr: list = None
  tokenizer: dict = None
  method: dict = None


@dataclass
class AnalysisConfigTemplate(BaseConfigTemplate):
  id: str
  requires: list
  create_features: list = None
  estimators: dict = None
  predict: dict = None
  plot: dict = None
  embedding_plot: dict = None