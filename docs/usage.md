# Usage Guide

## Quick Start

### Using the Gradio UI

1. **Start the UI**
```bash
python run_ui.py
```

2. **Initialize the Agent**
   - Enter your OpenAI API key
   - Click "Initialize Agent"

3. **Load Data**
   - Upload a CSV file OR
   - Generate synthetic data

4. **Ask Questions**
   - Use natural language queries
   - Click sample questions for examples

### Programmatic Usage

```python
from src.agent import WaferClusteringAgent
import pandas as pd

# Initialize agent
agent = WaferClusteringAgent(api_key="your-api-key")

# Load data
df = pd.read_csv("wafer_data.csv")
agent.load_data(df)

# Analyze
response = agent.analyze("Find optimal clusters")
print(response)
```

## Example Queries

### Data Exploration
- "What does my wafer data look like?"
- "Show me the distribution of yield values"
- "What features are available in the dataset?"

### Clustering Analysis
- "Find the optimal number of clusters"
- "Apply k-means clustering with 4 clusters"
- "Use DBSCAN to identify outliers"
- "Compare different clustering algorithms"

### Cluster Insights
- "What are the characteristics of each cluster?"
- "Which cluster has the highest yield?"
- "What factors differentiate the clusters?"

### Visualizations
- "Create a PCA visualization of the clusters"
- "Show me the feature distributions by cluster"
- "Plot yield vs defect density for each cluster"

## Data Format

Your CSV should include:

| Column | Description | Type |
|--------|-------------|------|
| Wafer_ID | Unique identifier | String |
| Yield_% | Yield percentage | Float |
| Defect_Density | Defects per area | Float |
| Temperature | Process temperature | Float |
| Pressure | Process pressure | Float |
| Process_Time | Duration in minutes | Float |
| ... | Other parameters | Float |

## Advanced Usage

### Custom Analysis Pipeline

```python
# 1. Load and explore data
agent.analyze("Describe the data")

# 2. Find optimal clusters
agent.analyze("Find optimal number of clusters using silhouette analysis")

# 3. Apply clustering
agent.analyze("Apply k-means with 3 clusters")

# 4. Analyze results
agent.analyze("What distinguishes each cluster?")

# 5. Identify issues
agent.analyze("Which wafers are outliers and why?")
```

### Comparing Algorithms

```python
# Compare multiple algorithms
queries = [
    "Apply k-means with 4 clusters",
    "Apply DBSCAN with default parameters",
    "Apply hierarchical clustering",
    "Which algorithm works best for this data?"
]

for query in queries:
    print(f"\n{query}")
    print(agent.analyze(query))
```

### Export Results

```python
# Get cluster assignments
df = agent.current_data
df['Cluster'] = agent.current_labels
df.to_csv('clustered_wafers.csv', index=False)
```

## Tips for Best Results

1. **Data Quality**
   - Ensure numeric columns are properly formatted
   - Handle missing values before analysis
   - Include relevant process parameters

2. **Query Formulation**
   - Be specific about what you want
   - Ask follow-up questions for deeper insights
   - Request visualizations for better understanding

3. **Interpretation**
   - Focus on business-relevant metrics (yield, defects)
   - Look for actionable patterns
   - Validate findings with domain knowledge

## Common Workflows

### Quality Control Workflow
1. Load production data
2. Identify outlier wafers
3. Analyze what makes them different
4. Get recommendations for process improvement

### Yield Optimization Workflow
1. Load historical data
2. Cluster by yield levels
3. Identify factors affecting yield
4. Find optimal process parameters

### Process Monitoring Workflow
1. Load recent production data
2. Compare with historical clusters
3. Detect process drift
4. Alert on anomalies