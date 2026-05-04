"""Convert 10_dynamis_v11_pheno_interval_ltae.py to a structured Jupyter notebook."""

import re
import json
import uuid
from pathlib import Path

SRC  = Path(r"C:\Users\eluzq\workspace\ai-for-good\notebooks\10_dynamis_v11_pheno_interval_ltae.py")
DEST = Path(r"C:\Users\eluzq\workspace\ai-for-good\notebooks\10_dynamis_v11_pheno_interval_ltae.ipynb")

source = SRC.read_text(encoding="utf-8")

# ── Cell splitting rules ────────────────────────────────────────────────────
# A new cell starts when we see one of these sentinel patterns at column 0:
CELL_SPLIT = re.compile(
    r"^(?:"
    r'# ─+|'          # section divider comments
    r'# ={3,}|'       # === dividers
    r'# -{3,}|'       # --- dividers
    r'class\s+\w+|'   # class definitions
    r'def\s+\w+|'     # top-level function definitions
    r'# In\[|'        # explicit cell markers
    r'# %%'           # percent-format cell markers
    r')',
    re.MULTILINE
)

# ── Parse into logical sections ─────────────────────────────────────────────
lines = source.splitlines(keepends=True)
cells_raw: list[list[str]] = [[]]

for line in lines:
    if CELL_SPLIT.match(line.rstrip('\n')):
        if any(l.strip() for l in cells_raw[-1]):  # flush non-empty cell
            cells_raw.append([])
    cells_raw[-1].append(line)

# ── Named sections for markdown headers ─────────────────────────────────────
SECTION_LABELS = {
    "# ─── Imports":                "## 📦 Imports",
    "# ─── Load Cache":             "## 💾 Load Cache",
    "# ─── Empirical Interval":     "## 📐 Empirical Interval Constants",
    "# ─── Build Data Arrays":      "## 🔧 Build Data Arrays",
    "# ─── Architecture":           "## 🏗️ Architecture",
    "# ─── Soft Cross-Entropy":     "## 🎯 Loss Function",
    "# ─── Train Crop Model":       "## 🌾 Train Crop Classification Model",
    "# ─── Train Rice Phenology":   "## 🌱 Train Rice Phenology Model",
    "# ─── Evaluation":             "## 📊 Evaluation",
    "# ─── Save Ensemble":          "## 💾 Save Ensemble",
}

def cell_to_markdown_header(lines_block: list[str]) -> str | None:
    """Return a markdown header string if the block starts with a known section."""
    first = ''.join(lines_block[:3]).strip()
    for sentinel, label in SECTION_LABELS.items():
        if sentinel in first:
            return label
    return None

# ── Build nbformat 4 notebook ────────────────────────────────────────────────
nb_cells = []

def make_md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }

def make_code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "execution_count": None,
        "metadata": {"collapsed": False},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }

# Title cell
title = "\n".join([
    "# 🛰️ Zero Hunger — V11: Phenophase Interval Temporal Anchor (Crop-Aware)",
    "",
    "**Key innovations over V10:**",
    "1. Corrected phenophase sequence: `Greenup→MidGreenup→Maturity→Peak→Senescence→MidSenescence→Dormancy`",
    "2. `PhenoIntervalEmbedding`: injects per-point observed intervals as temporal anchors",
    "3. Crop-specific interval priors as fallback (soybean senescence ~8.5d shorter than rice!)",
    "4. Between-region variation (SNR<1.0) captured by per-point observed intervals — no separate regional prior needed.",
    "",
    "> **Dataset:** 778 training points · 49 regions · 3 crops · verified 2026-05-03",
])
nb_cells.append(make_md(title))

for block in cells_raw:
    text = "".join(block).strip()
    if not text:
        continue

    md_header = cell_to_markdown_header(block)
    if md_header:
        nb_cells.append(make_md(md_header))

    nb_cells.append(make_code(text))

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "cells": nb_cells,
}

DEST.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Notebook written to: {DEST}")
print(f"   Total cells: {len(nb_cells)}")
