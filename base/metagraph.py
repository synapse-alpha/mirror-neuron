

class MetagraphMixin:
    
    @property
    def metagraph(self):
        print(f'Calling metagraph getter in {self.__class__.__name__} and self._metagraph is {self._metagraph}')
        assert self._metagraph is not None, 'metagraph not set. Please call set_metagraph()'
        return self._metagraph

    @metagraph.setter
    def metagraph(self, metagraph):
        print(f'Calling metagraph setter in {self.__class__.__name__} and self._metagraph is {self._metagraph}, it will be set to {metagraph}')
        self._metagraph = metagraph