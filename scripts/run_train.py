## Wrapper for mace.cli.run_train.main ##
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mace.cli.run_train import main

if __name__ == "__main__":
    # print(sys.path)
    main()
