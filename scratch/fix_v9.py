import json
import os

v9_path = r'C:\Users\eluzq\workspace\ai-for-good\notebooks\02_baseline_vs_dynamis_v9_amostra_audit.ipynb'

with open(v9_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'VERSION = 7' in source:
            new_source = source.replace('VERSION = 7', 'VERSION = 9')
            cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in new_source.split('\n')]
            if cell['source'][-1] == '\n': cell['source'].pop()

with open(v9_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Fixed VERSION in {v9_path}")
