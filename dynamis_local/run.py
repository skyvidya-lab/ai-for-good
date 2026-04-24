"""Entry point shortcut. Lets you run `python run.py <stage>` instead of
`python -m dynamis_local <stage>`. Functionally identical."""
import sys
from dynamis_local.cli import main

if __name__ == '__main__':
    sys.exit(main())
