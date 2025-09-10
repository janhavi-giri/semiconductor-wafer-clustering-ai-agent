# 🔬 Semiconductor Wafer Clustering AI Agent

An intelligent LangChain-based agent for analyzing semiconductor wafer data using advanced clustering techniques with a natural language interface.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/semiconductor-wafer-clustering-agent/blob/main/notebooks/Wafer_Clustering_Demo.ipynb)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 🤖 **Natural Language Interface** - Ask questions in plain English powered by GPT-4
- 📊 **Multiple Clustering Algorithms** - K-Means, DBSCAN, Hierarchical, GMM
- 📈 **Automatic Optimization** - Finds optimal number of clusters using silhouette and elbow methods
- 🎨 **Interactive Visualizations** - PCA plots, feature distributions, cluster comparisons
- 💻 **User-Friendly UI** - Gradio-based web interface for easy interaction
- 🔧 **Extensible Architecture** - Easy to add new tools and algorithms

![semiconductor-wafer-clustering-agent](https://github.com/janhavi-giri/semiconductor-wafer-clustering-agent/blob/main/WaferClusteringAIAgent-4August2025-gif.gif)


## 🚀 Quick Start

### Google Colab (Recommended)
The easiest way to get started is using Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/semiconductor-wafer-clustering-agent/blob/main/notebooks/Wafer_Clustering_Demo.ipynb)

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/janhavi-giri/semiconductor-wafer-clustering-ai-agent.git
cd semiconductor-wafer-clustering-ai-agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set your OpenAI API key**
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

4. **Run the UI**
```bash
python run_ui.py
```

## 📖 Usage

### Via Gradio UI

1. **Initialize the agent** with your OpenAI API key
2. **Load your data** - Upload CSV or generate synthetic data
3. **Ask questions** in natural language
4. **View results** - Get insights and visualizations

### Programmatic Usage

```python
from src.agent import WaferClusteringAgent

# Initialize agent
agent = WaferClusteringAgent(api_key="your-openai-api-key")

# Load your data
import pandas as pd
df = pd.read_csv("your_wafer_data.csv")
agent.load_data(df)

# Or generate synthetic data
df = agent.generate_synthetic_data(n_wafers=1000)
agent.load_data(df)

# Analyze using natural language
response = agent.analyze("Find the optimal number of clusters")
print(response)

response = agent.analyze("Apply k-means clustering and identify outliers")
print(response)
```

## 📊 Example Queries

- "What patterns exist in my wafer data?"
- "Find the optimal number of clusters for this dataset"
- "Apply k-means clustering with 4 clusters and analyze the results"
- "Which cluster has the highest yield?"
- "Compare k-means and DBSCAN clustering algorithms"
- "Identify any outlier wafers that need attention"
- "Create a PCA visualization of the clusters"
- "What factors correlate most strongly with wafer yield?"

## 📁 Data Format

Your CSV should contain wafer measurements with columns such as:
- `Wafer_ID` - Unique identifier
- `Yield_%` - Wafer yield percentage
- `Defect_Density` - Number of defects per unit area
- `Temperature` - Process temperature
- `Pressure` - Process pressure
- `Process_Time` - Processing duration
- Additional measurement parameters

## 🛠️ Requirements

- Python 3.8+
- OpenAI API key with GPT-4 access
- 8GB+ RAM recommended for large datasets

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Usage Guide](docs/usage.md)
- [API Reference](docs/api_reference.md)
- [Examples](examples/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/janhavi-giri/semiconductor-wafer-clustering-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/janhavi-giri/semiconductor-wafer-clustering-agent/discussions)

## 🙏 Acknowledgments

- Built with [LangChain](https://python.langchain.com/) for agent orchestration
- Powered by [OpenAI GPT-4](https://openai.com/) for natural language understanding
- UI created with [Gradio](https://gradio.app/)
- Clustering algorithms from [scikit-learn](https://scikit-learn.org/)

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{wafer_clustering_agent,
  title={Semiconductor Wafer Clustering AI Agent},
  author={Janhavi Giri},
  year={2025},
  url={https://github.com/janhavi-giri/semiconductor-wafer-clustering-ai-agent}
}
```

---

Made with ❤️ for the semiconductor industry
