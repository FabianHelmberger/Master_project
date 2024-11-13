# config.py

class Config:
    def __init__(self, **kwargs):
        self.steps = kwargs.get('steps', 1e3)
        self.dims = kwargs.get('dims', [10, 10])