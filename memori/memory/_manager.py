
import warnings

from memori._config import Config
from memori.memory._writer import Writer


class Manager:
    def __init__(self, config: Config):
        self.config = config

    def execute(self, payload):
        if self.config.enterprise is True:
            warnings.warn(
                "Memori Enterprise is not available yet.",
                RuntimeWarning,
                stacklevel=2,
            )

        Writer(self.config).execute(payload)

        return self
