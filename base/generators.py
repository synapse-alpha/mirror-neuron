"""
Also add:
- StepValue (constant value for a number of steps, parameterized by lo=0, hi=1, length=10, start=0)
- FrozenRandomValue (random value that is frozen after a number of steps, parameterized by lo=0, hi=1, length=10, start=0)
"""
import torch


class ConstantValue( torch.nn.Module ):

    def __init__(self, value):
        super(ConstantValue, self).__init__()
        self.value = value

    def forward(self, x, n, **kwargs):
        return torch.ones( n ) * self.value

class RandomValue( torch.nn.Module ):

    def __init__(self, seed=0, distribution='uniform', p0 = 1, p1 = 0, **kwargs):
        super(RandomValue, self).__init__()
        self.p0 = p0
        self.p1 = p1
        self.seed = seed
        self.distribution = distribution
        self.generator = torch.Generator().manual_seed(seed)
        
        if distribution == 'normal':
            self.sampler = torch.randn
        elif distribution == 'uniform':
            self.sampler = torch.rand
        else:
            raise ValueError(f'Unknown distribution {distribution}')

    def forward(self, x, n, **kwargs):
        return self.sampler( n, generator=self.generator ) * self.p0 + self.p1

class StepValue( torch.nn.Module ):

    def __init__(self, lo=0, hi=1, length=10, start=0, **kwargs):
        super(StepValue, self).__init__()
        self.lo = lo
        self.hi = hi
        self.length = length
        self.start = start

    def forward(self, x, n, **kwargs):
        return torch.linspace( self.lo, self.hi, self.length )[self.start:self.start+n]

class FrozenRandomValue( RandomValue ):

    def __init__(self, seed=0, distribution='uniform', p0 = 1, p1 = 0, **kwargs):
        super(FrozenRandomValue, self).__init__( seed=seed, distribution=distribution, p0=p0, p1=p1 )

    def forward(self, x, n, **kwargs):

        if not hasattr(self, 'frozen_values'):
            self.frozen_values = super(FrozenRandomValue, self).forward(x, n, **kwargs)

        return self.frozen_values