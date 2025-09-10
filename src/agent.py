"""
Main WaferClusteringAgent class for semiconductor wafer analysis
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI

from .tools import (
    DataInspectionTool,
    ClusteringTool,
    ClusterAnalysisTool,
    OptimalClustersTool,
    VisualizationTool
)
from .utils import shared_state


class WaferClusteringAgent:
    """
    Main agent for semiconductor wafer clustering analysis using LangChain
    
    This agent provides natural language interface for analyzing wafer data
    using various clustering algorithms and providing insights.
    """
    
    def __init__(self, openai_api_key: str):
        """
        Initialize the agent with OpenAI API key
        
        Args:
            openai_api_key: OpenAI API key for GPT-4 access
        """
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4",
            openai_api_key=openai_api_key
        )
        
        # Initialize tools
        self.tools = self._create_tools()
        
        # Create agent
        self.agent = self._create_agent()
        
    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent"""
        # Initialize tool instances
        tools = [
            DataInspectionTool(),
            ClusteringTool(),
            ClusterAnalysisTool(),
            OptimalClustersTool(),
            VisualizationTool()
        ]
        
        return [
            Tool(
                name=tool.name,
                func=tool._run,
                description=tool.description
            ) for tool in tools
        ]
    
    def _create_agent(self) -> AgentExecutor:
        """Create the ReAct agent"""
        
        # Create the prompt template
        prompt = PromptTemplate.from_template("""You are an expert semiconductor wafer analysis AI agent. Your goal is to help analyze wafer data using clustering techniques to identify patterns, anomalies, and insights.

Available tools:
{tools}

Tool Names: {tool_names}

When analyzing wafer data, follow these best practices:
1. Always inspect the data first to understand its structure
2. Find the optimal number of clusters before applying algorithms
3. Try multiple clustering algorithms and compare results
4. Analyze cluster characteristics to provide insights
5. Create visualizations to help understand patterns

IMPORTANT: 
- For apply_clustering tool, use JSON format like: {{"algorithm": "kmeans", "parameters": {{"n_clusters": 3}}}}
- The analyze_clusters tool requires no input - just use empty string ""
- For visualizations, specify type like: "pca_scatter" or "feature_distribution"

Use the following format:
Question: the input question you must answer
Thought: think about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer with insights and recommendations

Question: {input}
{agent_scratchpad}""")
        
        # Create agent executor
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            max_iterations=15
        )
    
    def load_data(self, data: pd.DataFrame):
        """
        Load wafer data into the agent
        
        Args:
            data: Pandas DataFrame containing wafer data
        """
        shared_state.wafer_data = data
        shared_state.scaled_data = None
        shared_state.current_labels = None
        shared_state.current_algorithm = None
        
        print(f"Loaded wafer data with shape: {data.shape}")
    
    def generate_synthetic_data(self, n_wafers: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic wafer data for testing
        
        Args:
            n_wafers: Number of wafers to generate
            
        Returns:
            DataFrame with synthetic wafer data
        """
        np.random.seed(42)
        
        # Generate different wafer types
        n_types = 4
        samples_per_type = n_wafers // n_types
        
        data_list = []
        
        for i in range(n_types):
            # Base features for each type
            base_yield = [95, 85, 75, 65][i]
            base_defects = [0.1, 0.5, 1.0, 2.0][i]
            
            type_data = {
                'Wafer_ID': [f'W{j:04d}' for j in range(i*samples_per_type, (i+1)*samples_per_type)],
                'Yield_%': np.random.normal(base_yield, 5, samples_per_type),
                'Defect_Density': np.random.exponential(base_defects, samples_per_type),
                'Temperature': np.random.normal(350 + i*10, 5, samples_per_type),
                'Pressure': np.random.normal(100 + i*5, 3, samples_per_type),
                'Process_Time': np.random.normal(120 + i*10, 10, samples_per_type),
                'Thickness': np.random.normal(500 + i*20, 15, samples_per_type),
                'Resistivity': np.random.normal(10 + i*2, 1, samples_per_type),
            }
            
            # Add more features
            for j in range(5):
                type_data[f'Param_{j+1}'] = np.random.normal(i*10, 5, samples_per_type)
            
            data_list.append(pd.DataFrame(type_data))
        
        # Combine and shuffle
        df = pd.concat(data_list, ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Clip values to reasonable ranges
        df['Yield_%'] = np.clip(df['Yield_%'], 0, 100)
        df['Defect_Density'] = np.clip(df['Defect_Density'], 0, None)
        
        return df
    
    def analyze(self, query: str) -> str:
        """
        Run analysis based on natural language query
        
        Args:
            query: Natural language question about the wafer data
            
        Returns:
            String response with analysis results
        """
        if shared_state.wafer_data is None:
            return "No data loaded. Please load wafer data first using load_data() method."
        
        # Run agent
        result = self.agent.invoke({"input": query})
        
        return result['output']
    
    @property
    def current_data(self) -> Optional[pd.DataFrame]:
        """Get the currently loaded data"""
        return shared_state.wafer_data
    
    @property
    def current_labels(self) -> Optional[np.ndarray]:
        """Get the current cluster labels"""
        return shared_state.current_labels
    
    @property
    def current_algorithm(self) -> Optional[str]:
        """Get the current clustering algorithm used"""
        return shared_state.current_algorithm