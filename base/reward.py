import bittensor
import torch

from base.values import ConstantValue, RandomValue
from abc import ABC, abstractmethod

# expose raw RewardModel for use in other modules
from sources.reward import RewardModel

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
# include query failure as an additional behavior
# TODO: inherit from RewardModel and just override init

class BaseRewardModel( torch.nn.Module, ABC ):

    def __init__(self, metagraph, **kwargs):
        super(BaseRewardModel, self).__init__()
        self._metagraph = metagraph
        #TODO: Hardcoded base tokenizer to facilitate code development. Make it dynamic to config in the future
        self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6b')
    
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, completions, rewards):
        pass
    
    @abstractmethod
    def reward(self, completions):
        pass


class DummyRewardModel( BaseRewardModel ):
    
    def __init__(self, reward_type='question_length', forward_value=ConstantValue(value=1), backward_value=ConstantValue(value=1), metagraph=None, **kwargs):
        super(DummyRewardModel, self).__init__( metagraph=metagraph )
        self.reward_type = reward_type
        self.forward_value = forward_value
        self.backward_value = backward_value

    def forward(self, x):
        # each neuron is given a score of 1
        return self.forward_type(x, self.metagraph.n.item())

    def backward(self, completions, rewards):
        # each neuron is given a score of 1
        return self.backward_value(completions, rewards, n=self.metagraph.n.item())

    def reward(self, completions):
        def reward_fn(samples):
            if self.reward_type == 'question_length':
                rewards = [len(msg) for msg in samples]
            elif self.reward_type == 'longest_word':
                rewards = [len(max(msg.split(), key=len)) for msg in samples]
            elif self.reward_type == 'num_words':
                rewards = [len(msg.split()) for msg in samples]
            return torch.tensor(rewards, dtype=torch.float32).mean()

        rewards = [reward_fn([completion]) for completion in completions]
        return torch.tensor(rewards, dtype=torch.float32)

class ConstantRewardModel( BaseRewardModel ):
    
    def __init__(self, forward_value=1, backward_value=0, metagraph=None, **kwargs):
        super(ConstantRewardModel, self).__init__( metagraph=metagraph )
        self.forward_value = ConstantValue(value=forward_value)
        self.backward_value = ConstantValue(value=backward_value)

    def forward(self, x):
        # each neuron is given a constant score
        return self.forward_value(x, n=1)

    def backward(self, completions, rewards, n=1):
        # each neuron is given a constant score
        return self.backward_value(completions, rewards, n=self.metagraph.n.item())

    def reward(self, completions):
        
        scores = [self.forward([completion]) for completion in completions]
        return torch.tensor(scores, dtype=torch.float32)

class RandomRewardModel( BaseRewardModel ):
    
    def __init__(self, seed=0, distribution='uniform', p0=1, p1=0, metagraph=None, **kwargs):
        super(RandomRewardModel, self).__init__( metagraph=metagraph )
        self.forward_value = RandomValue(seed=seed, distribution=distribution, p0=p0, p1=p1)
        self.backward_value = RandomValue(seed=seed, distribution=distribution, p0=kwargs.get('backward_p0',p0), p1=kwargs.get('backward_p1',p1))

    def forward(self, x):
        # each neuron is given a constant score
        return self.forward_value(x, n=1)

    def backward(self, completions, rewards):
        # each neuron is given a constant score
        return self.backward_value(completions, rewards, n=self.metagraph.n.item())

    def reward(self, completions):

        return torch.tensor([self.forward([completion]) for completion in completions], dtype=torch.float32)
    
class CustomRewardModel( RewardModel ):
    
    def __init__(self, model_path: str, device: str, config: 'bittensor.config' = None, **kwargs):
        super(CustomRewardModel, self).__init__(model_path, device, config)
        
        # just wrecklessly set all the kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)



class HuggingFaceRewardModel(BaseRewardModel):
    def __init__(
        self,
        model_path: str = "Dahoas/gpt2-rm-static",
        tokenizer_path: str = "EleutherAI/gpt-j-6b",
        device: str = "cpu",
        config: "bittensor.config" = None,  # NOTE: there is no such thing as a RewardModel.config attr defined in base.
        metagraph: "bittensor.metagraph" = None,
    ):
        super(HuggingFaceRewardModel, self).__init__(metagraph=metagraph)
        autoconfig = AutoConfig.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_config(autoconfig)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]
        self.config = self.model.config
        # NOTE: `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
        self.config.n_embd = (
            self.config.hidden_size
            if hasattr(self.config, "hidden_size")
            else self.config.n_embd
        )
        self.transformer = self.model.transformer
        self.v_head = torch.nn.Linear(self.config.n_embd, 1, bias=False)
        self.device = torch.device(device)
        self.model.to(self.device)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        labels=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        loss = None
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_end_scores = []
        rejected_end_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        chosen = input_ids[:bs]
        rejected = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        loss = 0
        inference = False
        for i in range(bs):
            if torch.all(torch.eq(chosen[i], rejected[i])).item():
                c_inds = (chosen[i] == self.PAD_ID).nonzero()
                c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
                chosen_end_scores.append(chosen_rewards[i, c_ind - 1])
                inference = True
                continue

            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
            r_inds = (rejected[i] == self.PAD_ID).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
            end_ind = max(c_ind, r_ind)

            # Retrieve first index where trajectories diverge
            divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
            assert divergence_ind > 0

            # Index into the correct rewards
            c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
            r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]

            # Append the last rewards to the list of end scores
            chosen_end_scores.append(c_truncated_reward[-1])
            rejected_end_scores.append(r_truncated_reward[-1])

            # Compute loss based on truncated rewards (ignore padding)
            loss += -torch.log(
                torch.sigmoid(c_truncated_reward - r_truncated_reward)
            ).mean()
        loss = loss / bs

        if not inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            rejected_end_scores = torch.stack(rejected_end_scores)

        if inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            return {"chosen_end_scores": chosen_end_scores}

        return {
            "loss": loss,
            "chosen_end_scores": chosen_end_scores,
            "rejected_end_scores": rejected_end_scores,
        }

    def reward(self, completions: List[str]) -> torch.FloatTensor:
        def reward_fn(samples):
            if samples is None:
                return 0
            scores_list = []
            batch_size = 1
            for i in range(0, len(samples), batch_size):
                sub_samples = samples[i : i + batch_size]
                sub_samples = [
                    "<|startoftext|>" + chosen + "<|endoftext|>"
                    for chosen in sub_samples
                ]
                encodings_dict = self.tokenizer(
                    sub_samples,
                    truncation=False,
                    max_length=550,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = encodings_dict["input_ids"].to(self.device)
                attn_masks = encodings_dict["attention_mask"].to(self.device)
                input_ids = input_ids.repeat(2, 1)
                attn_masks = attn_masks.repeat(2, 1)
                with torch.no_grad():
                    sub_scores = self.forward(
                        input_ids=input_ids, attention_mask=attn_masks
                    )
                scores_list.append(sub_scores["chosen_end_scores"])
            scores = torch.cat(scores_list, dim=0).mean().item()
            return scores

        with torch.no_grad():
            rewards = [reward_fn([completion]) for completion in completions]
            for completion, reward in zip(completions, rewards):
                print(completion)
                print(reward)
            return torch.tensor(rewards, dtype=torch.float32)
