r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__|_|
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                  [offline fork]
"""

import sys
from typing import Any

from memori._cli import Cli
from memori._config import Config
from memori._setup import Manager as SetupManager


def _cloud_feature_removed(name: str):
    """Factory for cloud feature stubs."""

    class RemovedFeatureManager:
        def __init__(self, config: Config):
            self.config = config

        def execute(self):
            cli = Cli(self.config)
            cli.notice(
                f"The '{name}' command has been removed in this offline fork.\n"
            )
            cli.notice(
                "Cloud features (quota, sign-up, cockroachdb cluster management) "
                "are no longer available.\n"
            )
            cli.notice(
                "This fork operates entirely locally with no connections to "
                "memorilabs.ai.\n"
            )
            sys.exit(1)

        def usage(self):
            print(f"The '{name}' command is no longer available (cloud feature removed)")

    return RemovedFeatureManager


def main():
    cli = Cli(Config())
    cli.banner()

    options: dict[str, dict[str, Any]] = {
        "cockroachdb": {
            "description": "[REMOVED] Cloud cluster management (offline fork)",
            "params": ["cluster", "<start | claim | delete>"],
            "obj": _cloud_feature_removed("cockroachdb cluster"),
        },
        "quota": {
            "description": "[REMOVED] Cloud quota check (offline fork)",
            "params": [],
            "obj": _cloud_feature_removed("quota"),
        },
        "setup": {
            "description": "Execute suggested setup steps",
            "params": [],
            "obj": SetupManager,
        },
        "sign-up": {
            "description": "[REMOVED] Cloud sign-up (offline fork)",
            "params": ["<email_address>"],
            "obj": _cloud_feature_removed("sign-up"),
        },
    }

    if len(sys.argv) <= 1 or sys.argv[1] not in options:
        cli.print("{:<15}{:<45}{:<6}".format("Option", "Description", "Params"))
        cli.print("{:<15}{:<45}{:<6}".format("------", "-----------", "------"))

        for key, value in options.items():
            params = value["params"]
            cli.print(
                "{:<15}{:<45}{:>6}".format(
                    key, value["description"], "Y" if len(params) > 0 else "N"
                )
            )

        cli.print("\nusage: python -m memori <option> [params]\n")
    else:
        option = options[sys.argv[1]]
        params = option["params"]
        obj_cls = option["obj"]
        if len(params) > 0:
            if len(sys.argv) != 2 + len(params):
                obj_cls(Config()).usage()
                cli.newline()
                sys.exit(1)

        obj_cls(Config()).execute()


if __name__ == "__main__":
    main()
