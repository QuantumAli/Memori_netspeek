r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__|_|
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                  [offline fork]
"""

import pyfiglet

from memori._config import Config


class Cli:
    def __init__(self, config: Config):
        self.config = config

    def banner(self):
        self.print(pyfiglet.figlet_format("Memori", font="standard").rstrip())
        self.print(" " * 18 + "perfectam memoriam")
        self.print(" " * 21 + "[offline fork]")
        self.print(" " * 30 + "v" + str(self.config.version) + "\n")

    def newline(self):
        self.print("")

    def notice(self, message, ident=0, end=None):
        prefix = "+ "
        if ident > 0:
            prefix = ""

        self.print(prefix + " " * (ident * 4) + message, end=end)

    def print(self, message, end=None):
        print(message, end=end, flush=True)
