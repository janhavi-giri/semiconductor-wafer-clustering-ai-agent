# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- OpenAI API key with GPT-4 access

## Installation Methods

### Method 1: Using pip (Recommended)

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/semiconductor-wafer-clustering-agent.git
cd semiconductor-wafer-clustering-agent
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Method 2: Using conda

```bash
# Create conda environment
conda create -n wafer-clustering python=3.9
conda activate wafer-clustering

# Install dependencies
pip install -r requirements.txt
```

### Method 3: Google Colab

No installation needed! Just open the notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/semiconductor-wafer-clustering-agent/blob/main/notebooks/Wafer_Clustering_Demo.ipynb)

## Setting up OpenAI API Key

### Option 1: Environment Variable

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

### Option 2: .env File

Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-your-api-key-here
```

### Option 3: Pass directly to the agent

```python
agent = WaferClusteringAgent(api_key="sk-your-api-key-here")
```

## Verify Installation

Run the following to verify everything is installed correctly:

```python
python -c "import langchain, gradio, sklearn, pandas; print('All packages installed successfully!')"
```

## Troubleshooting

### Issue: ModuleNotFoundError

Solution: Make sure you've activated your virtual environment and installed all requirements:
```bash
pip install -r requirements.txt
```

### Issue: OpenAI API Error

Solution: Verify your API key:
```python
import os
print(f"API Key set: {'OPENAI_API_KEY' in os.environ}")
```

### Issue: GPU/CUDA errors

The agent doesn't require GPU. If you see CUDA errors, they can be safely ignored.

## Next Steps

- See the [Usage Guide](usage.md) to start analyzing wafer data
- Check out the [Examples](../examples/) for sample code
- Run the UI with `python run_ui.py`