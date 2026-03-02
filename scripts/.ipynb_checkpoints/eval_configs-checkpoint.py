## Wrapper for mace.cli.eval_configs.main ##

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mace.cli.eval_configs import main

if __name__ == "__main__":
    main()
