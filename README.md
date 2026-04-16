# Dynamis Terra — AI for Good

Physics-informed crop type and phenophase classification from multi-temporal Sentinel-2 imagery.

Built for the **ITU AI and Space Computing Challenge 2026**, Track 1: *Space Intelligence Empowering Zero Hunger (SDG 2)*.

## 🚀 Project Overview

**Dynamis Terra** addresses the critical challenge of food security by improving the precision of crop monitoring. Our approach combines traditional physics-based crop phenology knowledge with modern machine learning (MKM - Markov-Kalman Module) to achieve robust results even with limited training data.

### Key Features
- **Markov-Kalman Module (MKM)**: Physics-informed rollout for state trajectory estimation.
- **ChaosAttention**: Lightweight attention mechanism with a 2-physics adapter (Chaos score + Hurst exponent).
- **Phenology Prior**: Initialises the state transition matrix with canonical crop growth phases.
- **Multimodal Classification**: Concurrent prediction of crop types (Rice, Corn, Soybean) and 7 phenophases.

## 📁 Repository Structure

- `src/dynamis/`: Core physics-informed modules (MKM, ChaosAttention).
- `src/data/`: Sentinel-2 data pipeline, vegetation indices, and temporal builders.
- `src/models/`: DynamisCropClassifier and configuration.
- `notebooks/`: Research and experiment notebooks (Colab-ready).
- `tests/`: Structural and logic sanity checks.

## 🛠 Setup & Installation

```bash
# Clone the repository
git clone https://github.com/GeoProjectAI/ai-for-good.git
cd ai-for-good

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
```

## 📓 Running in Google Colab

The project is designed to run seamlessly on Google Colab T4 GPUs:
1. Open `notebooks/00_colab_setup.ipynb` for environment preparation.
2. Run `notebooks/02_baseline_vs_dynamis.ipynb` for the main competition benchmarking.

## 🏆 Competition Details

- **Challenge**: ITU AI and Space Computing Challenge 2026.
- **Track**: Track 1 - Final Round (Zhejiang Lab / Zero2x).
- **Team**: Bonanza / Dynamis.

## 📄 License
Proprietary - Team Dynamis.
