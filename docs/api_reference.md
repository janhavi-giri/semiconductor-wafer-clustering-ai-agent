# API Reference

## Table of Contents
- [WaferClusteringAgent](#waferclusteringagent)
- [Tools](#tools)
  - [DataInspectionTool](#datainspectiontool)
  - [ClusteringTool](#clusteringtool)
  - [ClusterAnalysisTool](#clusteranalysistool)
  - [OptimalClustersTool](#optimalclusterstool)
  - [VisualizationTool](#visualizationtool)
- [UI Components](#ui-components)
- [Utility Functions](#utility-functions)

---

## WaferClusteringAgent

The main agent class that orchestrates the analysis of semiconductor wafer data using LangChain.

### Class: `WaferClusteringAgent`

```python
from src.agent import WaferClusteringAgent

agent = WaferClusteringAgent(openai_api_key: str)
```

#### Parameters:
- `openai_api_key` (str): Your OpenAI API key with GPT-4 access

#### Methods:

##### `load_data(data: pd.DataFrame) -> None`
Load wafer data into the agent for analysis.

```python
import pandas as pd
df = pd.read_csv("wafer_data.csv")
agent.load_data(df)
```

**Parameters:**
- `data`: Pandas DataFrame containing wafer measurements

**Expected columns:**
- `Wafer_ID`: Unique identifier (optional)
- Numeric columns for clustering (e.g., Yield_%, Defect_Density, Temperature, etc.)

##### `generate_synthetic_data(n_wafers: int = 1000) -> pd.DataFrame`
Generate synthetic wafer data for testing and demonstration.

```python
df = agent.generate_synthetic_data(n_wafers=500)
```

**Parameters:**
- `n_wafers`: Number of synthetic wafers to generate (default: 1000)

**Returns:**
- DataFrame with synthetic wafer data including:
  - Wafer_ID, Yield_%, Defect_Density
  - Temperature, Pressure, Process_Time
  - Thickness, Resistivity
  - Param_1 through Param_5

##### `analyze(query: str) -> str`
Analyze wafer data using natural language queries.

```python
response = agent.analyze("Find the optimal number of clusters")
print(response)
```

**Parameters:**
- `query`: Natural language question about the wafer data

**Returns:**
- String containing the analysis results

**Example queries:**
- "What patterns exist in my wafer data?"
- "Apply k-means clustering with 4 clusters"
- "Which wafers are outliers?"
- "Compare different clustering algorithms"

#### Properties:

##### `current_data -> Optional[pd.DataFrame]`
Get the currently loaded DataFrame.

```python
df = agent.current_data
```

##### `current_labels -> Optional[np.ndarray]`
Get the current cluster labels from the last clustering operation.

```python
labels = agent.current_labels
```

##### `current_algorithm -> Optional[str]`
Get the name of the last clustering algorithm used.

```python
algo = agent.current_algorithm  # e.g., "kmeans"
```

---

## Tools

LangChain tools used by the agent for specific operations.

### DataInspectionTool

Inspects and provides statistics about the loaded wafer data.

**Tool name:** `data_inspection`

**Input:** Empty string or any query (ignored)

**Output:** JSON containing:
- Data shape
- Column names and types
- Missing values count
- Statistical summary
- Sample rows

### ClusteringTool

Applies various clustering algorithms to the wafer data.

**Tool name:** `apply_clustering`

**Input:** JSON string with format:
```json
{
  "algorithm": "kmeans|dbscan|hierarchical|gmm",
  "parameters": {
    // algorithm-specific parameters
  }
}
```

**Algorithm parameters:**

#### K-Means
```json
{
  "algorithm": "kmeans",
  "parameters": {
    "n_clusters": 4
  }
}
```

#### DBSCAN
```json
{
  "algorithm": "dbscan",
  "parameters": {
    "eps": 1.5,
    "min_samples": 10
  }
}
```

#### Hierarchical
```json
{
  "algorithm": "hierarchical",
  "parameters": {
    "n_clusters": 4,
    "linkage": "ward"  // or "complete", "average", "single"
  }
}
```

#### Gaussian Mixture Model
```json
{
  "algorithm": "gmm",
  "parameters": {
    "n_components": 4
  }
}
```

**Output:** JSON containing:
- Algorithm used
- Parameters applied
- Number of clusters found
- Cluster sizes
- Performance metrics (silhouette score, Davies-Bouldin index, etc.)

### ClusterAnalysisTool

Analyzes the characteristics of each cluster.

**Tool name:** `analyze_clusters`

**Input:** Empty string (uses current clustering results)

**Output:** JSON containing:
- Cluster statistics for each feature
- Best/worst performing clusters
- Key insights

### OptimalClustersTool

Finds the optimal number of clusters using elbow and silhouette methods.

**Tool name:** `find_optimal_clusters`

**Input:** Maximum number of clusters to test (optional, default: 10)

**Output:** JSON containing:
- Optimal clusters by silhouette method
- Optimal clusters by elbow method
- Scores for each k value
- Recommendation

### VisualizationTool

Creates visualizations of the clustering results.

**Tool name:** `create_visualization`

**Input:** Visualization type as string:
- `"pca_scatter"`: PCA-based 2D scatter plot
- `"feature_distribution"`: Histograms by cluster
- `"cluster_comparison"`: Box plots comparing clusters

**Output:** Status message and saved plot file

---

## UI Components

### Class: `WaferClusteringUI`

Manages the Gradio interface state.

```python
from src.ui import WaferClusteringUI

ui = WaferClusteringUI()
```

#### Methods:

##### `initialize_agent(api_key: str) -> str`
Initialize the agent with API key.

##### `load_csv_data(file_obj) -> Tuple[str, str, str]`
Load data from uploaded CSV file.

##### `generate_synthetic_data(n_wafers: int) -> Tuple[str, str, str]`
Generate synthetic data.

##### `analyze_query(query: str, history: List) -> List`
Process natural language query.

##### `create_visualization(viz_type: str) -> Tuple[plt.Figure, str]`
Create standalone visualizations.

### Function: `create_gradio_interface() -> gr.Blocks`

Creates the complete Gradio interface.

```python
from src.ui import create_gradio_interface

demo = create_gradio_interface()
demo.launch(share=True)
```

---

## Utility Functions

### `convert_numpy_types(obj: Any) -> Any`

Convert NumPy types to native Python types for JSON serialization.

```python
from src.utils import convert_numpy_types

data = {"value": np.int64(42)}
json_safe = convert_numpy_types(data)
```

### `get_sample_queries() -> List[str]`

Get a list of example queries for the UI.

```python
from src.utils import get_sample_queries

queries = get_sample_queries()
```

### `validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]`

Validate that a DataFrame is suitable for wafer analysis.

```python
from src.utils import validate_dataframe

is_valid, message = validate_dataframe(df)
if not is_valid:
    print(f"Validation failed: {message}")
```

### Class: `SharedState`

Manages shared state between tools.

```python
from src.utils import shared_state

# Access shared data
wafer_data = shared_state.wafer_data
labels = shared_state.current_labels

# Reset state
shared_state.reset()
```

---

## Error Handling

The agent handles various error conditions:

- **No API Key**: Returns error message requesting valid key
- **No Data Loaded**: Prompts user to load data first
- **Invalid Query**: Provides helpful error messages
- **Clustering Failures**: Explains why clustering failed

Example error handling:

```python
try:
    response = agent.analyze("Complex query")
except Exception as e:
    print(f"Error: {e}")
    # Agent will provide helpful context
```

---

## Best Practices

1. **Data Preparation**
   - Ensure numeric columns for clustering
   - Handle missing values before loading
   - Include meaningful feature names

2. **Query Formulation**
   - Be specific about what you want
   - Mention algorithm names when needed
   - Ask follow-up questions for clarity

3. **Performance**
   - For large datasets (>10k wafers), consider sampling
   - Start with simple algorithms (k-means) before complex ones
   - Use optimal cluster detection before manual selection

4. **Interpretation**
   - Focus on business-relevant metrics (yield, defects)
   - Validate findings with domain knowledge
   - Export results for further analysis

---

## Examples

### Complete Analysis Pipeline

```python
# Initialize
agent = WaferClusteringAgent(api_key="your-key")

# Load data
df = pd.read_csv("wafers.csv")
agent.load_data(df)

# Full analysis pipeline
queries = [
    "Describe my wafer data",
    "Find optimal number of clusters",
    "Apply k-means with optimal clusters",
    "What distinguishes each cluster?",
    "Identify outliers using DBSCAN",
    "Create PCA visualization"
]

for query in queries:
    print(f"\nQ: {query}")
    print(f"A: {agent.analyze(query)}")

# Export results
if agent.current_labels is not None:
    df['Cluster'] = agent.current_labels
    df.to_csv('analyzed_wafers.csv')
```

### Custom Analysis Function

```python
def analyze_wafer_batch(agent, df, focus="yield"):
    """Custom analysis focusing on specific metric"""
    agent.load_data(df)
    
    # Optimal clusters
    optimal = agent.analyze("Find optimal clusters")
    
    # Apply clustering
    clustering = agent.analyze(f"Apply k-means with optimal clusters")
    
    # Focused analysis
    if focus == "yield":
        insights = agent.analyze(
            "Which cluster has highest yield? "
            "What process parameters distinguish it?"
        )
    elif focus == "defects":
        insights = agent.analyze(
            "Which clusters have high defect rates? "
            "What patterns do you see?"
        )
    
    return {
        "optimal_clusters": optimal,
        "clustering_results": clustering,
        "insights": insights,
        "labels": agent.current_labels
    }
```

---

## Extending the Agent

To add new tools or capabilities:

1. Create a new tool class inheriting from `BaseTool`
2. Implement the `_run` method
3. Add to agent's tool list
4. Update the prompt template if needed

Example:

```python
from langchain.tools import BaseTool

class YieldPredictionTool(BaseTool):
    name = "predict_yield"
    description = "Predict yield for new wafers"
    
    def _run(self, query: str) -> str:
        # Implementation
        return "Predictions..."
```