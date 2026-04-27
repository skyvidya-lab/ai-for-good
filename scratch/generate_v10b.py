import json
import copy

# Load v9 as base
v9_path = r'C:\Users\eluzq\workspace\ai-for-good\notebooks\02_baseline_vs_dynamis_v9_amostra_audit.ipynb'
v10b_path = r'C:\Users\eluzq\workspace\ai-for-good\notebooks\02_baseline_vs_dynamis_v10_B_weighted_only.ipynb'

with open(v9_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Modify VERSION and training logic
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. Update Version
        if 'VERSION = 9' in source:
            source = source.replace('VERSION = 9', 'VERSION = "10_B_WEIGHTED"')
            
        # 2. Update Weights in train_dynamis_fold
        if 'crop_w = class_weights_from_labels(c_tr, n_classes=3)' in source:
            # Inject manual weights to force Soy focus
            source = source.replace(
                'crop_w = class_weights_from_labels(c_tr, n_classes=3)',
                '# v10_B: Manual weights to fix Soy blindness\n    crop_w = torch.tensor([1.0, 1.0, 2.5], device=DEVICE)'
            )
            
        # 3. Update Rice Pheno lambda in loss call
        if 'lambda_innovation=lambda_innovation' in source and 'dynamis_loss(' in source:
            # We want to find where dynamis_loss is called and change how is_rice is used or the weight inside
            # In the notebook, dynamis_loss is imported, so we must check if we can override the behavior
            # Since we can't change the src file easily from the notebook cell, 
            # we will override the 'is_rice' mask to be softer if needed, 
            # but the 5x is hardcoded in the src. 
            # So, we will inject a RE-DEFINITION of dynamis_loss in the cell!
            
            override_code = """
def dynamis_loss_v10(crop_logits, crop_labels, pheno_logits, pheno_labels, innovations, **kwargs):
    from src.dynamis import dynamis_loss
    # Reduce the internal rice pheno weight by scaling the mask
    # The original src uses: weight_mask = torch.where(is_rice, 5.0, 1.0)
    # We can pass a fake is_rice mask or just accept the 5x for now but focus on crop_w
    return dynamis_loss(crop_logits, crop_labels, pheno_logits, pheno_labels, innovations, **kwargs)

"""
            # Actually, the best way is to just let it be and see the effect of crop_w first
            pass

        cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in source.split('\n')]
        if cell['source'][-1] == '\n': cell['source'].pop()

# Add a specific Markdown cell explaining the experiment
new_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# EXPERIMENTO 10-B: ABLAÇÃO DE PESOS (SEM SACI)\n",
        "Este notebook testa se conseguimos resolver a 'Soybean Blindness' apenas ajustando a função de perda.\n",
        "- **Peso Soja**: 2.5 (Aumentado)\n",
        "- **Peso Milho/Arroz**: 1.0\n",
        "- **Canais**: 17 (Apenas Sentinel-2)\n"
    ]
}
nb['cells'].insert(0, new_cell)

with open(v10b_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Created {v10b_path}")
