"""Command-line entry point.

Usage:
    python -m dynamis_local <stage> [--force] [--verbose]

Stages:
    extract   ZIP → EXTRACTED_DIR (idempotent)
    build     EXTRACTED_DIR → series_list cache
    baseline  Run LightGBM CV, save baseline_metrics.json
    dynamis   Run Dynamis CV + calibration + OOD + final model + checkpoint
    report    Assemble report.md from baseline + dynamis outputs
    all       Run all of the above in order

Flags:
    --force   Ignore caches / existing outputs and redo the stage.
    --verbose More print statements (currently unused — all stages verbose by default).

Before running, point DYNAMIS_ROOT at the folder with the 5 zips.
"""
from __future__ import annotations

import argparse
import sys
import traceback

from . import config, extract, build, baseline, dynamis_train, report


STAGES = {
    'extract': extract.run,
    'build': build.run,
    'baseline': baseline.run,
    'dynamis': dynamis_train.run,
    'report': report.run,
}

STAGE_ORDER = ['extract', 'build', 'baseline', 'dynamis', 'report']


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='python -m dynamis_local',
        description='Local training pipeline for Dynamis Terra (v8 audit).',
    )
    parser.add_argument(
        'stage',
        choices=['all', *STAGES.keys()],
        help='Which stage to run. `all` runs them in order.',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Ignore existing outputs / caches and redo the stage.',
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print extra diagnostics (reserved for future use).',
    )
    parser.add_argument(
        '--skip-validate', action='store_true',
        help='Skip the config validation (dangerous; only for debugging).',
    )
    args = parser.parse_args(argv)

    # Validate config (zips present, src importable) unless asked to skip
    if not args.skip_validate:
        try:
            config.validate()
        except Exception as e:
            print(f'[config] validation failed: {e}', file=sys.stderr)
            return 1

    # Dispatch
    stages_to_run = STAGE_ORDER if args.stage == 'all' else [args.stage]
    for stage in stages_to_run:
        print(f'\n========== {stage.upper()} ==========')
        try:
            STAGES[stage](force=args.force)
        except Exception as e:
            print(f'\n[{stage}] FAILED: {type(e).__name__}: {e}', file=sys.stderr)
            if args.verbose:
                traceback.print_exc()
            return 2
    print('\n========== DONE ==========')
    return 0


if __name__ == '__main__':
    sys.exit(main())
