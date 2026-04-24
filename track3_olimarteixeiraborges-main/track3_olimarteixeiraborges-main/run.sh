#!/bin/bash
python3 /workspace/scan_files.py
python3 /workspace/inference.py --data_dir /input --result_dir /output
