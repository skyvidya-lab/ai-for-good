import json
import sys

def py_to_ipynb(py_file, ipynb_file):
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content loosely by double newlines or major imports to make it a bit more readable
    # Actually, let's just make it a single large cell for simplicity, or split by some heuristics.
    # To keep it simple and ensure it works perfectly: 
    cells = []
    
    current_cell_lines = []
    for line in content.split('\n'):
        current_cell_lines.append(line + '\n')
    
    # Let's put everything in one cell, it's perfectly fine for Colab.
    # The user can split it if they want.
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": current_cell_lines
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    with open(ipynb_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

if __name__ == "__main__":
    py_to_ipynb('C:/Users/eluzq/workspace/ai-for-good/notebooks/09_dynamis_v10_top3_ltae.py', 'C:/Users/eluzq/workspace/ai-for-good/notebooks/09_dynamis_v10_top3_ltae.ipynb')
    print("Conversão concluída!")
