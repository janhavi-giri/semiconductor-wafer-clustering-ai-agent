"""
Gradio UI for the Semiconductor Wafer Clustering Agent
"""

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
import os

from .agent import WaferClusteringAgent
from .utils import get_sample_queries, shared_state


class WaferClusteringUI:
    """Gradio UI for Wafer Clustering Agent"""
    
    def __init__(self):
        self.agent = None
        self.current_data = None
        self.api_key = None
        self.chat_history = []
        
    def initialize_agent(self, api_key: str):
        """Initialize the clustering agent with API key"""
        try:
            if not api_key or api_key.strip() == "":
                return "❌ Please enter a valid API key"
            
            self.agent = WaferClusteringAgent(api_key)
            self.api_key = api_key
            return "✅ Agent initialized successfully!"
        except Exception as e:
            return f"❌ Error initializing agent: {str(e)}"
    
    def load_csv_data(self, file_path):
        """Load CSV data from uploaded file"""
        if file_path is None:
            return "Please upload a CSV file", None, None
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            self.current_data = df
            
            # Load data into agent if initialized
            if self.agent:
                self.agent.load_data(df)
            
            # Create data preview
            preview_html = df.head(10).to_html(index=False, classes="dataframe")
            
            # Create basic statistics
            stats_dict = {
                "Rows": len(df),
                "Columns": len(df.columns),
                "Numeric Columns": len(df.select_dtypes(include=[np.number]).columns),
                "Missing Values": df.isnull().sum().sum()
            }
            stats_df = pd.DataFrame(list(stats_dict.items()), columns=["Metric", "Value"])
            stats_html = stats_df.to_html(index=False, classes="dataframe")
            
            return f"✅ Data loaded successfully! Shape: {df.shape}", preview_html, stats_html
            
        except Exception as e:
            return f"❌ Error loading data: {str(e)}", None, None
    
    def generate_synthetic_data(self, n_wafers: int):
        """Generate synthetic wafer data"""
        try:
            if self.agent is None:
                return "Please initialize the agent first", None, None
            
            # Generate data
            df = self.agent.generate_synthetic_data(n_wafers=n_wafers)
            self.current_data = df
            self.agent.load_data(df)
            
            # Create preview
            preview_html = df.head(10).to_html(index=False, classes="dataframe")
            
            # Create statistics
            stats_dict = {
                "Rows": len(df),
                "Columns": len(df.columns),
                "Features": ", ".join([col for col in df.columns if col != 'Wafer_ID'][:5]) + "..."
            }
            stats_df = pd.DataFrame(list(stats_dict.items()), columns=["Metric", "Value"])
            stats_html = stats_df.to_html(index=False, classes="dataframe")
            
            return f"✅ Generated {n_wafers} synthetic wafers!", preview_html, stats_html
            
        except Exception as e:
            return f"❌ Error generating data: {str(e)}", None, None
    
    def analyze_query(self, query: str, history: List[Tuple[str, str]]):
        """Process natural language query through the agent"""
        if self.agent is None:
            return history + [(query, "❌ Please initialize the agent with your API key first")]
        
        if self.current_data is None:
            return history + [(query, "❌ Please load data first (upload CSV or generate synthetic data)")]
        
        try:
            # Add loading message
            history = history + [(query, "🔄 Processing...")]
            
            # Get response from agent
            response = self.agent.analyze(query)
            
            # Update history with actual response
            history[-1] = (query, response)
            
            return history
            
        except Exception as e:
            history[-1] = (query, f"❌ Error: {str(e)}")
            return history
    
    def create_visualization(self, viz_type: str):
        """Create standalone visualizations"""
        if self.current_data is None:
            return None, "Please load data first"
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if viz_type == "Yield Distribution":
                if 'Yield_%' in self.current_data.columns:
                    self.current_data['Yield_%'].hist(bins=30, ax=ax, edgecolor='black')
                    ax.set_xlabel('Yield %')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Wafer Yield Distribution')
                else:
                    ax.text(0.5, 0.5, 'Yield_% column not found', ha='center', va='center')
                    
            elif viz_type == "Correlation Matrix":
                numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns[:10]
                corr_matrix = self.current_data[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
                ax.set_title('Feature Correlation Matrix')
                
            elif viz_type == "Defect vs Yield":
                if 'Yield_%' in self.current_data.columns and 'Defect_Density' in self.current_data.columns:
                    ax.scatter(self.current_data['Defect_Density'], 
                              self.current_data['Yield_%'], alpha=0.5)
                    ax.set_xlabel('Defect Density')
                    ax.set_ylabel('Yield %')
                    ax.set_title('Yield vs Defect Density')
                else:
                    ax.text(0.5, 0.5, 'Required columns not found', ha='center', va='center')
                    
            elif viz_type == "Feature Statistics":
                numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns[:5]
                self.current_data[numeric_cols].boxplot(ax=ax)
                ax.set_title('Feature Statistics (Box Plot)')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            return fig, f"✅ Created {viz_type} visualization"
            
        except Exception as e:
            return None, f"❌ Error creating visualization: {str(e)}"


def create_gradio_interface():
    """Create and configure the Gradio interface"""
    ui = WaferClusteringUI()
    
    with gr.Blocks(title="Wafer Clustering AI Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔬 Semiconductor Wafer Clustering AI Agent
        
        An intelligent assistant for analyzing semiconductor wafer data using advanced clustering techniques.
        Powered by LangChain and GPT-4.
        """)
        
        with gr.Tab("🔧 Setup"):
            gr.Markdown("### Step 1: Initialize the Agent")
            
            with gr.Row():
                api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    placeholder="sk-...",
                    type="password",
                    scale=3
                )
                init_btn = gr.Button("Initialize Agent", variant="primary", scale=1)
            
            init_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("### Step 2: Load Your Data")
            
            with gr.Tab("Upload CSV"):
                file_upload = gr.File(
                    label="Upload Wafer Data (CSV)",
                    file_types=[".csv"],
                    type="filepath"
                )
                upload_btn = gr.Button("Load CSV Data", variant="primary")
                
            with gr.Tab("Generate Synthetic"):
                n_wafers_slider = gr.Slider(
                    minimum=100,
                    maximum=5000,
                    value=1000,
                    step=100,
                    label="Number of Wafers"
                )
                generate_btn = gr.Button("Generate Synthetic Data", variant="primary")
            
            load_status = gr.Textbox(label="Load Status", interactive=False)
            
            with gr.Row():
                data_preview = gr.HTML(label="Data Preview")
                data_stats = gr.HTML(label="Data Statistics")
        
        with gr.Tab("💬 Chat Analysis"):
            gr.Markdown("""
            ### Ask Questions About Your Wafer Data
            Use natural language to analyze patterns, apply clustering, and get insights.
            """)
            
            chatbot = gr.Chatbot(
                label="Conversation",
                height=400,
                bubble_full_width=False,
                type="tuples"
            )
            
            with gr.Row():
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., Find the optimal number of clusters for my wafer data",
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            # Sample queries using buttons
            gr.Markdown("#### Quick Questions:")
            sample_queries = get_sample_queries()
            
            # Create buttons for sample queries
            with gr.Row():
                sample_btns = []
                for i, query in enumerate(sample_queries[:4]):  # First 4 queries
                    btn = gr.Button(query, size="sm", scale=1)
                    sample_btns.append((btn, query))
            
            with gr.Row():
                for i, query in enumerate(sample_queries[4:]):  # Remaining queries
                    btn = gr.Button(query, size="sm", scale=1)
                    sample_btns.append((btn, query))
            
            clear_btn = gr.Button("Clear Conversation")
        
        with gr.Tab("📊 Visualizations"):
            gr.Markdown("### Quick Visualizations")
            
            with gr.Row():
                viz_type = gr.Dropdown(
                    choices=[
                        "Yield Distribution",
                        "Correlation Matrix",
                        "Defect vs Yield",
                        "Feature Statistics"
                    ],
                    value="Yield Distribution",
                    label="Visualization Type"
                )
                create_viz_btn = gr.Button("Create Visualization", variant="primary")
            
            viz_plot = gr.Plot(label="Visualization")
            viz_status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Tab("📖 Help"):
            gr.Markdown("""
            ### How to Use This Interface
            
            1. **Initialize the Agent**: Enter your OpenAI API key and click "Initialize Agent"
            2. **Load Data**: Either upload a CSV file or generate synthetic wafer data
            3. **Ask Questions**: Use natural language to analyze your data
            4. **Visualize**: Create quick visualizations of your data
            
            ### Example Questions:
            - "What patterns exist in my wafer data?"
            - "Apply k-means clustering with 4 clusters"
            - "Which wafers are outliers?"
            - "Compare different clustering algorithms"
            - "What factors correlate with high yield?"
            
            ### Data Format:
            Your CSV should contain wafer measurements with columns like:
            - Wafer_ID
            - Yield_%
            - Defect_Density
            - Temperature, Pressure, Process_Time
            - Other measurement parameters
            
            ### Troubleshooting:
            - If you get an "Error" message, make sure you've initialized the agent and loaded data
            - The agent needs both steps completed before analyzing
            - Check that your API key is valid and has access to GPT-4
            
            ### About:
            This tool uses advanced clustering algorithms to analyze semiconductor wafer data
            and identify patterns that can help improve manufacturing processes.
            """)
        
        # Event handlers
        init_btn.click(
            fn=ui.initialize_agent,
            inputs=[api_key_input],
            outputs=[init_status]
        )
        
        upload_btn.click(
            fn=ui.load_csv_data,
            inputs=[file_upload],
            outputs=[load_status, data_preview, data_stats]
        )
        
        generate_btn.click(
            fn=ui.generate_synthetic_data,
            inputs=[n_wafers_slider],
            outputs=[load_status, data_preview, data_stats]
        )
        
        submit_btn.click(
            fn=ui.analyze_query,
            inputs=[query_input, chatbot],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",
            outputs=[query_input]
        )
        
        query_input.submit(
            fn=ui.analyze_query,
            inputs=[query_input, chatbot],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",
            outputs=[query_input]
        )
        
        # Set up sample query button clicks
        for btn, query_text in sample_btns:
            btn.click(
                fn=lambda q=query_text: q,
                outputs=[query_input]
            )
        
        clear_btn.click(
            fn=lambda: [],
            outputs=[chatbot]
        )
        
        create_viz_btn.click(
            fn=ui.create_visualization,
            inputs=[viz_type],
            outputs=[viz_plot, viz_status]
        )
    
    return demo