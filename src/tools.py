"""
LangChain tools for wafer clustering analysis
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from .utils import shared_state, convert_numpy_types


class DataInspectionTool(BaseTool):
    """Tool for inspecting wafer data"""
    name: str  = "data_inspection"
    description: str = "Inspect the structure and statistics of wafer data. Use this to understand the dataset."
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Inspect the loaded wafer data"""
        if shared_state.wafer_data is None:
            return "No data loaded. Please load wafer data first."
        
        df = shared_state.wafer_data
        
        # Get basic info
        info = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
            "missing_values": convert_numpy_types(df.isnull().sum().to_dict()),
            "statistics": convert_numpy_types(df.describe().to_dict()),
            "sample_rows": convert_numpy_types(df.head(3).to_dict())
        }
        
        return json.dumps(info, indent=2, default=str)


class ClusteringTool(BaseTool):
    """Tool for applying clustering algorithms"""
    name: str = "apply_clustering"
    description: str = """Apply a clustering algorithm to wafer data. 
    Input should be a JSON string with keys: 
    - algorithm: 'kmeans', 'dbscan', 'hierarchical', or 'gmm'
    - parameters: dict of algorithm-specific parameters
    Example: {"algorithm": "kmeans", "parameters": {"n_clusters": 3}}"""
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Apply clustering algorithm"""
        try:
            # Clean the input string
            query = query.strip()
            if query.startswith("```json"):
                query = query[7:]
            if query.startswith("```"):
                query = query[3:]
            if query.endswith("```"):
                query = query[:-3]
            query = query.strip()
            
            # Parse input
            try:
                params = json.loads(query)
            except json.JSONDecodeError:
                # Try to parse as simple format
                if "kmeans" in query.lower():
                    params = {"algorithm": "kmeans", "parameters": {"n_clusters": 3}}
                elif "dbscan" in query.lower():
                    params = {"algorithm": "dbscan", "parameters": {"eps": 1.5, "min_samples": 10}}
                elif "hierarchical" in query.lower():
                    params = {"algorithm": "hierarchical", "parameters": {"n_clusters": 3}}
                elif "gmm" in query.lower():
                    params = {"algorithm": "gmm", "parameters": {"n_components": 3}}
                else:
                    return "Error: Could not parse input. Please use JSON format."
            
            algorithm = params.get('algorithm', 'kmeans')
            algo_params = params.get('parameters', {})
            
            # Preprocess data if needed
            if shared_state.scaled_data is None:
                df = shared_state.wafer_data
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                shared_state.scaler = StandardScaler()
                shared_state.scaled_data = shared_state.scaler.fit_transform(df[numerical_cols])
            
            # Apply clustering based on algorithm
            if algorithm == 'kmeans':
                n_clusters = algo_params.get('n_clusters', 3)
                model = KMeans(n_clusters=n_clusters, random_state=42)
            elif algorithm == 'dbscan':
                eps = algo_params.get('eps', 1.5)
                min_samples = algo_params.get('min_samples', 10)
                model = DBSCAN(eps=eps, min_samples=min_samples)
            elif algorithm == 'hierarchical':
                n_clusters = algo_params.get('n_clusters', 3)
                linkage = algo_params.get('linkage', 'ward')
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            elif algorithm == 'gmm':
                n_components = algo_params.get('n_components', 3)
                model = GaussianMixture(n_components=n_components, random_state=42)
            else:
                return f"Unknown algorithm: {algorithm}"
            
            # Fit and predict
            labels = model.fit_predict(shared_state.scaled_data)
            
            # Store results in shared state
            shared_state.current_labels = labels
            shared_state.current_algorithm = algorithm
            
            # Calculate metrics
            metrics = {}
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            if n_clusters > 1:
                metrics['silhouette_score'] = silhouette_score(shared_state.scaled_data, labels)
                if -1 not in labels:  # Exclude DBSCAN noise points
                    metrics['davies_bouldin_score'] = davies_bouldin_score(shared_state.scaled_data, labels)
                    metrics['calinski_harabasz_score'] = calinski_harabasz_score(shared_state.scaled_data, labels)
            
            # Get cluster sizes
            cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()
            
            result = {
                "algorithm": algorithm,
                "parameters": algo_params,
                "n_clusters": n_clusters,
                "n_samples": len(labels),
                "metrics": metrics,
                "cluster_sizes": convert_numpy_types(cluster_sizes),
                "status": "success"
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error applying clustering: {str(e)}"


class ClusterAnalysisTool(BaseTool):
    """Tool for analyzing cluster characteristics"""
    name: str = "analyze_clusters"
    description: str = "Analyze the characteristics of clusters including yield, defects, and other parameters. No input required."
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Analyze cluster characteristics"""
        if shared_state.current_labels is None:
            return "No clustering results found. Please apply clustering first."
        
        df = shared_state.wafer_data.copy()
        df['Cluster'] = shared_state.current_labels
        
        analysis = {
            "algorithm_used": shared_state.current_algorithm,
            "total_samples": len(df),
            "clusters": {}
        }
        
        # Analyze each cluster
        for cluster in sorted(df['Cluster'].unique()):
            if cluster == -1:  # Skip noise points for DBSCAN
                continue
                
            cluster_data = df[df['Cluster'] == cluster]
            cluster_info = {
                "size": len(cluster_data),
                "percentage": f"{len(cluster_data) / len(df) * 100:.1f}%",
                "statistics": {}
            }
            
            # Focus on key features
            key_features = ['Yield_%', 'Defect_Density', 'Temperature', 'Pressure', 'Process_Time']
            
            for feature in key_features:
                if feature in cluster_data.columns:
                    cluster_info["statistics"][feature] = {
                        "mean": round(cluster_data[feature].mean(), 3),
                        "std": round(cluster_data[feature].std(), 3),
                        "min": round(cluster_data[feature].min(), 3),
                        "max": round(cluster_data[feature].max(), 3)
                    }
            
            analysis["clusters"][f"Cluster_{cluster}"] = cluster_info
        
        # Add insights
        if 'Yield_%' in df.columns and 'Defect_Density' in df.columns:
            # Find best and worst clusters by yield
            cluster_yields = {}
            for cluster in df['Cluster'].unique():
                if cluster != -1:
                    cluster_yields[cluster] = df[df['Cluster'] == cluster]['Yield_%'].mean()
            
            if cluster_yields:
                best_cluster = max(cluster_yields, key=cluster_yields.get)
                worst_cluster = min(cluster_yields, key=cluster_yields.get)
                
                analysis["insights"] = {
                    "best_yield_cluster": best_cluster,
                    "best_yield_value": round(cluster_yields[best_cluster], 2),
                    "worst_yield_cluster": worst_cluster,
                    "worst_yield_value": round(cluster_yields[worst_cluster], 2),
                    "yield_difference": round(cluster_yields[best_cluster] - cluster_yields[worst_cluster], 2)
                }
        
        return json.dumps(analysis, indent=2, default=str)


class OptimalClustersTool(BaseTool):
    """Tool for finding optimal number of clusters"""
    name: str = "find_optimal_clusters"
    description: str = "Find the optimal number of clusters using elbow method and silhouette analysis. Input: max_clusters (optional, default=10)"
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Find optimal number of clusters"""
        try:
            if query.strip():
                max_clusters = int(query.strip())
            else:
                max_clusters = 10
        except:
            max_clusters = 10
        
        if shared_state.scaled_data is None:
            df = shared_state.wafer_data
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            shared_state.scaler = StandardScaler()
            shared_state.scaled_data = shared_state.scaler.fit_transform(df[numerical_cols])
        
        inertias = []
        silhouette_scores = []
        K = list(range(2, min(max_clusters + 1, len(shared_state.scaled_data) // 10)))
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(shared_state.scaled_data)
            inertias.append(float(kmeans.inertia_))
            silhouette_scores.append(float(silhouette_score(shared_state.scaled_data, kmeans.labels_)))
        
        # Find optimal k (highest silhouette score)
        optimal_k = int(K[np.argmax(silhouette_scores)])
        
        # Calculate elbow point
        if len(K) > 2:
            # Simple elbow detection: find point with maximum curvature
            deltas = np.diff(inertias)
            delta_deltas = np.diff(deltas)
            elbow_idx = np.argmax(np.abs(delta_deltas)) + 2  # +2 because of double diff
            elbow_k = int(min(elbow_idx, len(K)))
        else:
            elbow_k = int(K[0])
        
        result = {
            "optimal_clusters_silhouette": optimal_k,
            "optimal_clusters_elbow": elbow_k,
            "silhouette_scores": {str(k): round(score, 4) for k, score in zip(K, silhouette_scores)},
            "inertias": {str(k): round(inertia, 2) for k, inertia in zip(K, inertias)},
            "recommendation": f"Silhouette suggests {optimal_k} clusters, elbow suggests {elbow_k} clusters"
        }
        
        return json.dumps(result, indent=2)


class VisualizationTool(BaseTool):
    """Tool for creating visualizations"""
    name: str = "create_visualization"
    description: str = """Create visualizations of clustering results. 
    Input should specify type: 'pca_scatter', 'cluster_comparison', or 'feature_distribution'"""
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Create visualization"""
        viz_type = query.strip().lower()
        
        if shared_state.current_labels is None:
            return "No clustering results to visualize. Please apply clustering first."
        
        try:
            if 'pca' in viz_type or 'scatter' in viz_type:
                # PCA visualization
                pca = PCA(n_components=2)
                data_2d = pca.fit_transform(shared_state.scaled_data)
                
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], 
                                    c=shared_state.current_labels, cmap='viridis', 
                                    alpha=0.6, edgecolors='black', linewidth=0.5)
                plt.colorbar(scatter)
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                plt.title(f'{shared_state.current_algorithm.upper()} Clustering Results (PCA Visualization)')
                plt.grid(True, alpha=0.3)
                
                # Add cluster centers
                for cluster in set(shared_state.current_labels):
                    if cluster != -1:  # Skip noise
                        cluster_points = data_2d[shared_state.current_labels == cluster]
                        center = cluster_points.mean(axis=0)
                        plt.scatter(center[0], center[1], c='red', s=200, marker='x', 
                                  linewidths=3, label=f'Cluster {cluster} center' if cluster == 0 else "")
                
                plt.savefig('cluster_pca.png', dpi=150, bbox_inches='tight')
                plt.show()
                
                return "PCA scatter plot created and saved as 'cluster_pca.png'"
                
            elif 'feature' in viz_type or 'distribution' in viz_type:
                # Feature distribution by cluster
                df = shared_state.wafer_data.copy()
                df['Cluster'] = shared_state.current_labels
                
                # Select important features
                features = ['Yield_%', 'Defect_Density'] 
                if 'Yield_%' not in df.columns:
                    features = list(df.select_dtypes(include=[np.number]).columns[:2])
                
                fig, axes = plt.subplots(1, len(features), figsize=(15, 5))
                if len(features) == 1:
                    axes = [axes]
                
                colors = plt.cm.viridis(np.linspace(0, 1, len(set(shared_state.current_labels))))
                
                for i, feature in enumerate(features):
                    for j, cluster in enumerate(sorted(df['Cluster'].unique())):
                        if cluster != -1:  # Skip noise
                            cluster_data = df[df['Cluster'] == cluster][feature]
                            axes[i].hist(cluster_data, alpha=0.6, label=f'Cluster {cluster}', 
                                       bins=20, color=colors[j], edgecolor='black')
                    
                    axes[i].set_xlabel(feature)
                    axes[i].set_ylabel('Frequency')
                    axes[i].set_title(f'{feature} Distribution by Cluster')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('feature_distribution.png', dpi=150, bbox_inches='tight')
                plt.show()
                
                return "Feature distribution plots created and saved as 'feature_distribution.png'"
                
            elif 'comparison' in viz_type:
                # Cluster comparison plot
                df = shared_state.wafer_data.copy()
                df['Cluster'] = shared_state.current_labels
                
                # Create box plots for key features
                features = ['Yield_%', 'Defect_Density', 'Temperature', 'Pressure']
                available_features = [f for f in features if f in df.columns]
                
                if not available_features:
                    available_features = list(df.select_dtypes(include=[np.number]).columns[:4])
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.ravel()
                
                for i, feature in enumerate(available_features[:4]):
                    df.boxplot(column=feature, by='Cluster', ax=axes[i])
                    axes[i].set_title(f'{feature} by Cluster')
                    axes[i].set_xlabel('Cluster')
                    axes[i].set_ylabel(feature)
                
                plt.suptitle('Cluster Comparison', fontsize=16)
                plt.tight_layout()
                plt.savefig('cluster_comparison.png', dpi=150, bbox_inches='tight')
                plt.show()
                
                return "Cluster comparison plots created and saved as 'cluster_comparison.png'"
                
            else:
                return f"Unknown visualization type: {viz_type}. Available: 'pca_scatter', 'feature_distribution', 'cluster_comparison'"
                
        except Exception as e:
            return f"Error creating visualization: {str(e)}"