import json
import os
from pathlib import Path

v9_path = r'C:\Users\eluzq\workspace\ai-for-good\notebooks\02_baseline_vs_dynamis_v9_amostra_audit.ipynb'
v10_path = r'C:\Users\eluzq\workspace\ai-for-good\notebooks\02_baseline_vs_dynamis_v10_agro_saci.ipynb'

with open(v9_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update VERSION and titles
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'VERSION = 7' in source or 'VERSION = 9' in source:
            new_source = source.replace('VERSION = 7', 'VERSION = 10').replace('VERSION = 9', 'VERSION = 10')
            new_source = new_source.replace('# v7: all-region sample', '# v10: agroclimate enrichment (SACI)')
            cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in new_source.split('\n')]
            if cell['source'][-1] == '\n': cell['source'].pop()
    
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if 'v7 \u2014 All-Region Sample' in source or 'baseline vs Dynamis' in source:
            new_source = source.replace('v7 \u2014 All-Region Sample', 'v10 \u2014 Agroclimate Enrichment (SACI)')
            new_source = new_source.replace('v6 Audit Foundation', 'v9 Audit + SACI GEE')
            cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in new_source.split('\n')]
            if cell['source'][-1] == '\n': cell['source'].pop()

# Add SACI loading logic after Cell 3 (Feature Pipeline)
# Find the cell that builds series_list
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'series_list.append(ps)' in "".join(cell['source']):
        # This is the cell. We should add a new cell after it to load SACI
        saci_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# v10 \u2014 SACI Agroclimate Enrichment\n",
                "from src.data.agroclimate_enrichment import AgroclimateExtractor\n",
                "import pickle\n",
                "\n",
                "SACI_PATH = f'{WORKSPACE}/saci_enrichment_v9.pkl'\n",
                "if os.path.exists(SACI_PATH):\n",
                "    print(f'[saci] Loading agroclimate data from {SACI_PATH}')\n",
                "    with open(SACI_PATH, 'rb') as f:\n",
                "        saci_data = pickle.load(f)\n",
                "    \n",
                "    extractor = AgroclimateExtractor(project='agente-bdr-sdr')\n",
                "    \n",
                "    enriched_count = 0\n",
                "    for ps in series_list:\n",
                "        if ps.point_id in saci_data:\n",
                "            agro_df = saci_data[ps.point_id]\n",
                "            aligned = extractor.align_to_ps(ps.dates, agro_df)\n",
                "            # Append the 4 agro channels to features: (T, 17) -> (T, 21)\n",
                "            ps.features = np.concatenate([ps.features, aligned.values], axis=1)\n",
                "            enriched_count += 1\n",
                "    \n",
                "    print(f'[saci] Enriched {enriched_count} points with 4 agroclimate channels.')\n",
                "    N_FEATURES_ENRICHED = 21\n",
                "else:\n",
                "    print(f'[saci] WARNING: {SACI_PATH} not found. Skipping enrichment.')\n",
                "    N_FEATURES_ENRICHED = 17\n"
            ]
        }
        nb['cells'].insert(i + 1, saci_cell)
        break

# Update X construction to use N_FEATURES_ENRICHED
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'X = np.full((n_points, T_max, N_FEATURES)' in "".join(cell['source']):
        source = "".join(cell['source'])
        new_source = source.replace('N_FEATURES', 'N_FEATURES_ENRICHED')
        cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in new_source.split('\n')]
        if cell['source'][-1] == '\n': cell['source'].pop()

# Update Model input dimension
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'DynamisCropClassifier(' in "".join(cell['source']):
        source = "".join(cell['source'])
        new_source = source.replace('in_channels=N_FEATURES', 'in_channels=N_FEATURES_ENRICHED')
        # Also add weighted loss
        if 'nn.CrossEntropyLoss(' in new_source:
            new_source = new_source.replace(
                'nn.CrossEntropyLoss()', 
                'nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 2.0]).to(DEVICE))'
            )
        cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in new_source.split('\n')]
        if cell['source'][-1] == '\n': cell['source'].pop()

with open(v10_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Created {v10_path}")
