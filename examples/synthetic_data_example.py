#!/usr/bin/env python3
"""
Example: Using the Wafer Clustering Agent with Synthetic Data

This example demonstrates how to:
1. Generate synthetic wafer data
2. Perform clustering analysis
3. Get insights using natural language
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import WaferClusteringAgent

def main():
    # Initialize agent
    print("Initializing Wafer Clustering Agent...")
    
    # You can set your API key here or use environment variable
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    agent = WaferClusteringAgent(api_key)
    
    # Generate synthetic data
    print("\nGenerating synthetic wafer data...")
    df = agent.generate_synthetic_data(n_wafers=1000)
    print(f"Generated {len(df)} wafers with {len(df.columns)} features")
    
    # Load data into agent
    agent.load_data(df)
    
    # Example analysis workflow
    queries = [
        "What does the wafer data look like?",
        "Find the optimal number of clusters for this dataset",
        "Apply k-means clustering with the optimal number of clusters",
        "What are the key characteristics of each cluster?",
        "Which cluster has the best yield? What makes it different?",
        "Are there any outlier wafers I should investigate?",
        "Create a visualization showing the cluster separation"
    ]
    
    print("\n" + "="*60)
    print("Starting Analysis")
    print("="*60)
    
    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 60)
        
        try:
            response = agent.analyze(query)
            print(response)
        except Exception as e:
            print(f"Error: {e}")
        
        print()
    
    # Save results
    print("\nSaving clustered data...")
    results_df = df.copy()
    if hasattr(agent, 'current_labels') and agent.current_labels is not None:
        results_df['Cluster'] = agent.current_labels
        results_df.to_csv('synthetic_wafers_clustered.csv', index=False)
        print("Results saved to 'synthetic_wafers_clustered.csv'")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()