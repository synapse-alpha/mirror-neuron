# reward_model_analysis

Experiments on bittensor reward models to analyze behaviour and find exploits. This package is structured in a general way so that it can be used for tracking other models such as the gating model. It is also intended to be extensible so that extra components can be tested in future versions.

# Setup
Create a virtual environment
```bash
python3 -m venv env
```

Source virtual environment
```bash
source env/bin/activate
```

Install dependencies
```bash
pip install -r requirements.txt
```

**Requires bittensor==4.0.0**
```bash
pip list | grep bittensor
```

Login to weights and biases
```bash
wandb login
```

# Run

Uses yaml config files to define experiment and tracks results using [weights and biases](https://wandb.ai/site)

To run an experiment:

```bash
python3 main --config <file_name>
```

## Config File

An experiment consists of several steps which are all defined in the **config file**. An experiment may be run on a single step or a subset of steps.
1. [Load reward model](#load-reward-model)
2. [Load data](#load-data)
3. [Query model](#query-model)
4. [Analyze responses](#analyze-reponses)

### Load Reward Model
The reward model is defined in the config file in the following way:
```yaml
model:
  id: my_reward_model
  path: EleutherAI/gpt-j-6b
  ckpt: https://huggingface.co/Dahoas/gptj-rm-static/resolve/main/hf_ckpt.pt
  tokenizer: default
  embedding: default
  norm: default
```
  entrypoint: RewardModel

### Load Data
The data is defined in the config file in the following way:
```yaml
data:
  id: my_data
  path: nq_open
  schema:
    train:
      - row:
        question: question
        answer: answer
  sample: default
    method: first
    n: 1000
```

### Query Model
The query is defined in the config file in the following way:

```yaml
query:
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
```

### Analyze Responses
The analysis is defined in the config file in the following way:

```yaml
analyze:
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
    - gbr:
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
    - loss:
      x: embedding
      estimator: gbr
    - chosen_end_scores:
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
```


