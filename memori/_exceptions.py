r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__|_|
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                  [offline fork]
"""

import warnings
from importlib.metadata import PackageNotFoundError, distribution


class ExtractorNotConfiguredError(Exception):
    """Raised when augmentation is attempted without a configured extractor."""

    def __init__(
        self,
        message=(
            "No augmentation extractor configured. "
            "This offline fork requires a local extractor for advanced augmentation. "
            "Configure one using:\n"
            "  from memori.memory.augmentation.extractors import GroqExtractor\n"
            "  memori.config.augmentation_extractor = GroqExtractor()"
        ),
    ):
        self.message = message
        super().__init__(self.message)


class MemoriLegacyPackageWarning(UserWarning):
    """Warning emitted when the legacy `memorisdk` package is installed."""


def warn_if_legacy_memorisdk_installed() -> None:
    try:
        distribution("memorisdk")
    except PackageNotFoundError:
        return

    warnings.warn(
        "You have Memori installed under the legacy package name 'memorisdk'. "
        "That name is deprecated and will stop receiving updates. "
        "Please switch to 'memori':\n\n"
        "    pip uninstall memorisdk\n"
        "    pip install memori\n",
        MemoriLegacyPackageWarning,
        stacklevel=3,
    )
