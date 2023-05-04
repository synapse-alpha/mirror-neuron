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

Also the code performance can be profiled using `pyinstrument` by passing an additional command line argument
```bash
python3 main --config dummy_config.yml --profile
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
    dendrite_pool:
      name: DummyDendritePool
      args:
        fail_rate: 0.1
        data_path: nq_open
    gating_model:
      name: RandomGatingModel
      args:
        frozen: True
        seed: 0
        distribution: uniform
    reward_model:
      name: RandomRewardModel
      args:
        forward_value: 1
        backward_value: 0
    alpha: 0.01
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
    sample:
      method: first
      n: 100
```

### Query Model
The query is defined in the config file in the following way:

```yaml
query:
    id: my_query
    chunk_size: 1
    method:
      # name: forward
      name: train
      args:
        max_iter: 10
    ignore_attr:
      - hotkeys
      - block

```

### Analyze Responses
The analysis is defined in the config file in the following way:

```yaml
analysis:
    id: my_analysis
    requires:
      - my_query
    create_features:
      - question_length
      - num_words
      - avg_word_length
    plot:
      rewards:
        - question_length
        - num_words
        - avg_word_length
      scores:
        - question_length
        - num_words
        - avg_word_length       
```


