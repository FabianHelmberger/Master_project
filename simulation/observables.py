# simulation/observables.py

class Observables:
    def __init__(self, config, field):
        self.config = config
        self.field = field
        self.results = []

    def compute(self):
        # Compute observables based on the current state of the field
        observable_value = ...  # Some calculation
        self.results.append(observable_value)
