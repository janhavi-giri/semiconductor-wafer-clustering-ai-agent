#!/usr/bin/env python3
"""
Example: Analyzing CSV Wafer Data

This example shows how to:
1. Load wafer data from CSV
2. Perform comprehensive analysis
3. Export results
"""

import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import WaferClusteringAgent

def main():
    # Initialize agent
    print("Initializing Wafer Clustering Agent...")
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    agent = WaferClusteringAgent(api_key)
    
    # Load CSV data
    csv_file = "sample_data.csv"  # Change this to your CSV file
    
    try:
        print(f"\nLoading data from {csv_file}...")
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} wafers with columns: {list(df.columns)}")
        
        # Load into agent
        agent.load_data(df)
        
    except FileNotFoundError:
        print(f"Error: {csv_file} not found!")
        print("Using sample data instead...")
        
        # Create sample data if file not found
        sample_data = {
            'Wafer_ID': [f'W{i:04d}' for i in range(100)],
            'Yield_%': pd.np.random.normal(85, 10, 100),
            'Defect_Density': pd.np.random.exponential(0.5, 100),
            'Temperature': pd.np.random.normal(350, 10, 100),
            'Pressure': pd.np.random.normal(100, 5, 100),
            'Process_Time': pd.np.random.normal(120, 15, 100)
        }
        df = pd.DataFrame(sample_data)
        agent.load_data(df)
    
    # Comprehensive analysis workflow
    print("\n" + "="*60)
    print("Starting Comprehensive Wafer Analysis")
    print("="*60)
    
    # 1. Data Exploration
    print("\n1. DATA EXPLORATION")
    print("-" * 40)
    response = agent.analyze(
        "Analyze the wafer data and provide summary statistics. "
        "What are the key features and their distributions?"
    )
    print(response)
    
    # 2. Optimal Clustering
    print("\n2. FINDING OPTIMAL CLUSTERS")
    print("-" * 40)
    response = agent.analyze(
        "Find the optimal number of clusters using both elbow method "
        "and silhouette analysis. Explain your recommendation."
    )
    print(response)
    
    # 3. Apply Clustering
    print("\n3. APPLYING CLUSTERING")
    print("-" * 40)
    response = agent.analyze(
        "Apply k-means clustering with the optimal number of clusters. "
        "Then also try DBSCAN to see if there are any outliers."
    )
    print(response)
    
    # 4. Cluster Analysis
    print("\n4. CLUSTER CHARACTERISTICS")
    print("-" * 40)
    response = agent.analyze(
        "Analyze the characteristics of each cluster. "
        "What makes each cluster unique? Focus on yield and defects."
    )
    print(response)
    
    # 5. Business Insights
    print("\n5. BUSINESS INSIGHTS")
    print("-" * 40)
    response = agent.analyze(
        "Based on the clustering results, what are the key insights "
        "for improving wafer production? Which wafers need attention?"
    )
    print(response)
    
    # 6. Visualization
    print("\n6. CREATING VISUALIZATIONS")
    print("-" * 40)
    response = agent.analyze(
        "Create a PCA visualization of the clusters and explain "
        "what patterns you see in the data."
    )
    print(response)
    
    # Export results
    print("\n" + "="*60)
    print("Exporting Results")
    print("="*60)
    
    if hasattr(agent, 'current_labels') and agent.current_labels is not None:
        # Add cluster labels to data
        results_df = df.copy()
        results_df['Cluster'] = agent.current_labels
        
        # Calculate cluster statistics
        cluster_stats = results_df.groupby('Cluster').agg({
            'Yield_%': ['mean', 'std', 'count'],
            'Defect_Density': ['mean', 'std']
        }).round(2)
        
        # Save files
        results_df.to_csv('wafer_analysis_results.csv', index=False)
        cluster_stats.to_csv('cluster_statistics.csv')
        
        print("✅ Saved wafer_analysis_results.csv")
        print("✅ Saved cluster_statistics.csv")
        print("\nCluster Statistics:")
        print(cluster_stats)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()