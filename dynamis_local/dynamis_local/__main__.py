"""Entry point so `python -m dynamis_local` works."""
from .cli import main
import sys

if __name__ == '__main__':
    sys.exit(main())
