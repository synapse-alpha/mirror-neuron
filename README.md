# Mirror Neuron

Creates **tracked experiments on bittensor text prompting models** to analyze behaviour and find exploits. 

The goal of this repo is to enable systematic study of the interaction between components of the network and their effect on network performance.  


<picture>
  <img alt="Shows an illustrated sun in light mode and a moon with stars in dark mode." src="https://user-images.githubusercontent.com/6709103/235761418-c305c8ee-dc84-46fb-851c-e2cf3b074366.png" width=400>
</picture>



An experiment contains 4 steps:
1. **Model**: Define model (at present this must be a text prompting validator neuron)
2. **Data**: Define data to query the model
3. **Query**: Query the model and generate response data 
4. **Analysis**: Analyze the response data 

The custom `Neuron` model has following controllable components:
1. **Dendrite Pool** is replaced with `DummyDendritePool` which imitates the behaviour of the miners 
2. **Gating Model** can be replaced with baseline models such as `RandomGatingModel`, `ConstantGatingModel` or more complex implementations (TBD)
3. **Reward Model** can also be replaced with baseline models such as `RandomRewardModel`, `ConstantRewardModel` or more complex implementations (TBD)

The custom `Neuron` instance has a modified entrypoint `__init__` function, but it can be interfaced with using the *same API as bittensor* in order to examine the behaviour of the system (`.forward()`, `.train()`, `.inference()`) in the **query** step of an experiment.

**Note**: This repo explicitly copies and modifies the source code of all files in `neurons/text/prompting/validator/core/` so that they are more configurable for experimentation purposes. It would be ideal if a future version of bittensor reflects a similarly configurable API so that no source code needs to be copied into this repo.

# Setup
Create a virtual environment
```bash
python3 -m venv ~/.mirror
```

Source virtual environment
```bash
source ~/.mirror/bin/activate
```

Install bittensor
```bash
git clone -b text_prompting --single-branch https://github.com/opentensor/bittensor.git
cd bittensor
python -m pip install .
cd ..
```

Install mirror_neuron dependencies
```
git clone https://github.com/opentensor/mirror_neuron.git
cd mirror_neuron
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

Each run requires a yaml config file to define the experiment and track results using [weights and biases](https://wandb.ai/site)

To run an experiment:

```bash
python3 main --config <file_name>
```

To test locally (and not spam wandb)

```bash
python3 main --config <file_name> --offline
```

## Example Usage
Start with the dummy configuration to ensure everything works smoothly. 
```bash
python3 main --config dummy_config.yml
```

## Config File

An experiment consists of several steps which are all defined in the **config file**. An experiment may be run on a single step or a subset of steps.
1. [Load model](#load-model)
2. [Load data](#load-data)
3. [Query model](#query-model)
4. [Analyze responses](#analyze-reponses)

### Load  Model
The model is defined in the config file in the following way:
```yaml
model:
  id: my_model
  path: EleutherAI/gpt-j-6b
  ckpt: https://huggingface.co/Dahoas/gptj-rm-static/resolve/main/hf_ckpt.pt
  tokenizer: default
  embedding: default
  norm: default
```

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
analysis:
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


