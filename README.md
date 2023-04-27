# reward_model_analysis

Experiments on bittensor reward models to analyze behaviour and find exploits. This package is structured in a general way so that it can be used for tracking other models such as the gating model. It is also intended to be extensible so that extra components can be tested in future versions.

# Setup
Create a virtual environment
`python3 -m venv env`

Source virtual environment
`source env/bin/activate`

Install dependencies
`pip install -r requirements.txt`

**Requires bittensor==4.0.0**
`pip list | grep bittensor`

# Run

Uses yaml config files to define experiment and tracks results using [weights and biases](https://wandb.ai/site)

## Experiment Steps
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
      - row
        question: question
        answer: answer
  sample: default
    method: first
    n: 1000
```

### Query Model
```yaml
model:
  id: my_query
  path: EleutherAI/gpt-j-6b
  ckpt: https://huggingface.co/Dahoas/gptj-rm-static/resolve/main/hf_ckpt.pt
  tokenizer: default
  embedding: default
  norm: default
```
  entrypoint: RewardModel

### Analyze Responses



