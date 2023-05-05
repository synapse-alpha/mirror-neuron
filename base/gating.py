import torch
import bittensor
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import List, Union

from base.values import ConstantValue, RandomValue, FrozenRandomValue
from abc import ABC, abstractmethod

# expose raw GatingModel for use in other modules
from sources.gating import GatingModel
from sources.neuron import neuron
# TODO: inherit from GatingModel and just override init


class BaseGatingModel( torch.nn.Module, ABC ):

    def __init__(self, metagraph):
        super(BaseGatingModel, self).__init__()
        self._metagraph = metagraph

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, completions, rewards):
        pass


class LongestMessageGatingModel( BaseGatingModel ):

    def __init__(self, metagraph=None):
        super(LongestMessageGatingModel, self).__init__( )
        self._metagraph = metagraph

    def forward(self, x, n):
        # return ones for the longest n messages and zeros for the rest
        threshold = torch.topk(x, n, largest=False).values[-1]
        return torch.ones( x.shape ) * (x >= threshold).float()

    def backward(self, scores, rewards):

        return torch.ones( self.metagraph.n.item() )


class RandomGatingModel( BaseGatingModel ):

    def __init__(self, frozen=False, seed=0, distribution='uniform', p0=1, p1=0, metagraph=None, config: 'bittensor.config' = None):
        super(RandomGatingModel, self).__init__( metagraph=metagraph )

        value_type = FrozenRandomValue if frozen else RandomValue
        self.value = value_type(seed=seed, distribution=distribution, p0=p0, p1=p1)

    def forward(self, x):
        # each neuron is given a random score
        return self.value(x, self.metagraph.n.item())

    def backward(self, scores, rewards):
        # each neuron is given a random score
        return self.value(torch.zeros( self.metagraph.n.item() ), self.metagraph.n.item())


class ConstantGatingModel( BaseGatingModel ):

    def __init__(self, value=1, metagraph=None):
        super(ConstantGatingModel, self).__init__( metagraph=metagraph )
        self.value = ConstantValue(value)

    def forward(self, x):
        # each neuron is given a constant score
        return self.value(x, self.metagraph.n.item())

    def backward(self, scores, rewards):

        # each neuron is given a random score
        return torch.random( self.metagraph.n.item() )


class MaskedGatingModel( BaseGatingModel ):

    def __init__(self, mask=10, metagraph=None):
        super(MaskedGatingModel, self).__init__( metagraph=metagraph )
        raise NotImplementedError(f'Not implemented yet.')

class SequentialGatingModel( BaseGatingModel ):

    def __init__(self, metagraph: 'bittensor.metagraph.Metagraph' = None, config: 'bittensor.config' = None, tokenizer_name: str = None, hidden_size: Union[int, List[int]] = None, embedding_dim: int = None, num_uids: int = None):
        """
        Initializes the SequentialGatingModel. (mostly copied from GatingModel)
        - `metagraph`: A reference to the Bittensor metagraph object.
        - `config`: Configuration object for the gating model. If `None`, the default configuration is used.
        - `tokenizer_name`: Name of the pre-trained transformer-based language model to use as the encoding layer for the gating model. If `None`, the default model name specified in the configuration is used.
        - `hidden_size`: Size of the hidden layers of the gating model. If `None`, the default size specified in the configuration is used.
        - `embedding_dim`: Dimension of the embedding layer of the gating model. If `None`, the default size specified in the configuration is used.
        - `num_uids`: Number of uids to gate on. If `None`, the default number specified in the configuration is used.
        """
        super(SequentialGatingModel, self).__init__( metagraph=metagraph )
        if config is None: 
            config = neuron.config()
        if tokenizer_name is not None:
            config.gating.tokenizer_name = tokenizer_name
        if num_uids is not None: 
            config.gating.num_uids = num_uids
        if embedding_dim is not None:
            config.gating.embedding_dim = embedding_dim
            
        # 1. sentence is passed into forward method: 'this is a question' batch_size x sentence_length
        # 2. sentence is tokenized: [12, 54, 39, 90] batch_size x n_tokens
        # 3. tokens are passed into embedding layer: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.10, 0.11, 0.12]] batch_size x n_tokens x embedding_size
        # 4. pass embedded tokens into gating model: [score1, score2, score3, score4] num_uids
            
        self.config = config
        self.metagraph = metagraph
        self.device = torch.device( self.config.neuron.device )
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.gating.tokenizer_name)
        self.tokenizer.model_max_length = 128
        # Make hidden size a list if it is not already
        self.hidden_size = hidden_size if isinstance(hidden_size, list) else [hidden_size]
        
        # Embedding layer (not trainable)
        self.embedding_dim = self.config.gating.embedding_dim
        self.embedding_layer = torch.nn.Embedding(num_embeddings=self.tokenizer.vocab_size, embedding_dim=self.embedding_dim)
        self.embedding_layer.weight.requires_grad = False
        
        # LSTM layers
        self.layers = [torch.nn.LSTM(self.embedding_dim, self.hidden_size[0], batch_first=True)]
        self.layers += [torch.nn.LSTM(self.hidden_size[i], hidden_dim,  batch_first=True) for i, hidden_dim in enumerate(self.hidden_size[1:])]

        self.model = torch.nn.Sequential(*([self.embedding_layer]+self.layers))
        # generalize this so that num_hidden can be zero/empty so that we just learn weights from embedding space to uid space
        
        # Linear layer
        self.linear = torch.nn.Linear(self.hidden_size[-1], self.metagraph.n)
        self.optimizer = torch.optim.SGD(
            [ {"params": self.parameters()} ],
            lr = self.config.gating.learning_rate,
            momentum = self.config.gating.momentum,
        )
        
    def forward( self, message: str ) -> 'torch.FloatTensor':
        """ Runs a forward pass through the model.
            Args:
                message (:obj:`str`): 
                    text message to be encoded.
            Returns:
                scores (:obj:`torch.FloatTensor` of shape :obj:`(network_size)`):
                    Scores for each uids as output by the gating model.
        """
        inputs = self.tokenizer( message, return_tensors="pt" ).to( self.device )
        hidden_states = self.model( inputs['input_ids'] )#.last_hidden_state[0, -1, :]
        return self.linear( hidden_states[0] )[0,-1,:]


    def backward( self, scores: torch.FloatTensor, rewards: torch.FloatTensor ): 
        """ Runs a backward pass through the model.
            Args:
                scores (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    Scores for each uids as output by the gating model.
                rewards (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    Rewards for each uids as output by the reward model.
        """   
        normalized_scores = torch.nn.functional.softmax( scores, dim=0 ).to( self.device )
        normalized_rewards = torch.nn.functional.softmax( rewards, dim=0 ).to( self.device )
        loss = torch.nn.functional.mse_loss( normalized_scores, normalized_rewards.detach() )
        loss.backward()
        self.optimizer.step()