"""
Utility functions and shared classes for the Wafer Clustering Agent
"""

import numpy as np
from typing import Any, Dict, List, Union

def convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy types to Python native types for JSON serialization
    
    Args:
        obj: Object to convert
        
    Returns:
        Converted object with native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    else:
        return obj

class SharedState:
    """
    Manages shared state between tools
    
    This class maintains the state that needs to be shared across
    different LangChain tools, including data, scaling information,
    and clustering results.
    """
    
    def __init__(self):
        """Initialize empty shared state"""
        self.wafer_data = None
        self.scaled_data = None
        self.current_labels = None
        self.current_algorithm = None
        self.scaler = None
        
    def reset(self):
        """Reset all state variables"""
        self.wafer_data = None
        self.scaled_data = None
        self.current_labels = None
        self.current_algorithm = None
        self.scaler = None
        
    def has_data(self) -> bool:
        """Check if data is loaded"""
        return self.wafer_data is not None
        
    def has_clustering(self) -> bool:
        """Check if clustering has been performed"""
        return self.current_labels is not None

# Global state instance
shared_state = SharedState()

def get_sample_queries() -> List[str]:
    """
    Get a list of sample queries for the UI
    
    Returns:
        List of example questions users can ask
    """
    return [
        "Analyze the wafer data and tell me what features are available",
        "Find the optimal number of clusters for this dataset",
        "Apply k-means clustering with 3 clusters and analyze the results",
        "What are the characteristics of each cluster?",
        "Which clustering algorithm works best for this data?",
        "Identify any outlier wafers",
        "Create a PCA visualization of the clusters",
        "Compare k-means and DBSCAN clustering results"
    ]

def validate_dataframe(df) -> tuple[bool, str]:
    """
    Validate that a dataframe is suitable for wafer analysis
    
    Args:
        df: Pandas DataFrame to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    if df is None:
        return False, "No data provided"
        
    if len(df) < 10:
        return False, "Dataset too small. Need at least 10 wafers."
        
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return False, "Need at least 2 numeric columns for clustering"
        
    if df.isnull().sum().sum() > len(df) * len(df.columns) * 0.5:
        return False, "Too many missing values (>50%)"
        
    return True, "Data validation passed"