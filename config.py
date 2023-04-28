
"""
TEMPLATE
  id: my_reward_model
  path: EleutherAI/gpt-j-6b
  ckpt: https://huggingface.co/Dahoas/gptj-rm-static/resolve/main/hf_ckpt.pt
  tokenizer: default
  embedding: default
  norm: default
"""
model_config_template = {
    'id': {'type': str, 'required': True},
    'path': {'type': str, 'required': True},
    'ckpt': {'type': str, 'required': False, 'default': None},
    'tokenizer': {'type': str, 'required': False, 'default': None},
    'embedding': {'type': str, 'required': False, 'default': None},
    'norm': {'type': str, 'required': False, 'default': None},
}

"""
DATA TEMPLATE
data:
  id: my_data
  path: nq_open
  schema:
    train:
      - row:
        question: question
        answer: answer
  sample: 
    method: first
    n: 1000
"""
data_config_template = {
    'id': {'type': str, 'required': True},
    'path': {'type': str, 'required': True},
    'schema': {'type': dict, 'required': True},
    'sample': {'type': dict, 'required': False, 'default': None}
}

"""
QUERY TEMPLATE
 id: my_query
  requires:
    - my_reward_model
    - my_data
  entrypoint: RewardModel
  chunk_size: 1
  message:
    roles: default
    unravel: default
  tokenizer:
    truncation: False
    max_length: 550
    padding: max_length
  forward:
    inference: False
    args:
        input_ids: None
        past_key_values: None
        attention_mask: None
        token_type_ids: None
        position_ids: None
        head_mask: None
        inputs_embeds: None
        mc_token_ids: None
        labels: None
        return_dict: True
        output_attentions: False
        output_hidden_states: True
  store:
    - loss
    - chosen_end_scores
    - rejected_end_scores
    - hidden_states
    - tokens
    - embeddings

"""
query_config_template = {
    'id': {'type': str, 'required': True},
    'requires': {'type': list, 'required': True},
    'entrypoint': {'type': str, 'required': True},
    'chunk_size': {'type': int, 'required': False, 'default': 1},
    'message': {'type': dict, 'required': False, 'default': None},
    'tokenizer': {'type': dict, 'required': False, 'default': None},
    'forward': {'type': dict, 'required': False, 'default': None},
    'store': {'type': list, 'required': False, 'default': None}
}

"""
ANALYSIS TEMPLATE
  id: my_analysis
  requires:
    - my_query
  compute:
    - question_length
    - num_words
    - avg_word_length
    - embedding:
      id: my_sentence_embedding
      type: sentence
      path: sentence-transformers/all-MiniLM-L6-v2
  estimators:
    gbr:
      name: GradientBoostingRegressor
      args:
        n_estimators: 100
        min_samples_leaf: 5
        validation_fraction: 0.1
        n_iter_no_change: 10
        random_state: 0
        verbose: 1
        subsample: 0.9
  predict:
    loss:
      x: embedding
      estimator: gbr
    chosen_end_scores:
      x: embedding
      estimator: gbr
  plot:
    - loss:
      - question_length
      - num_words
      - avg_word_length
    - chosen_end_scores:
      - question_length
      - num_words
      - avg_word_length
    - rejected_end_scores:
      - question_length
      - num_words
      - avg_word_length
"""

analysis_config_template = {
    'id': {'type': str, 'required': True},
    'requires': {'type': list, 'required': True},
    'compute': {'type': dict, 'required': False, 'default': None},
    'estimators': {'type': dict, 'required': False, 'default': None},
    'predict': {'type': dict, 'required': False, 'default': None},
    'plot': {'type': list, 'required': False, 'default': None}
}