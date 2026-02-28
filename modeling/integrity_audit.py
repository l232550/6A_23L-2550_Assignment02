import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def plot_regional_integrity(df, use_sequential=True):
    """Task 4: Small multiples showing Pollution vs Zone vs Country."""
    # Aggregate data: average PM2.5 by country and zone
    df_agg = df.groupby(['country', 'zone']).agg({
        'pm25': ['mean', 'std', 'count']
    }).reset_index()
    df_agg.columns = ['country', 'zone', 'pm25_mean', 'pm25_std', 'count']
    
    # Create faceted bar chart (small multiples)
    color_scale = 'viridis' if use_sequential else 'rainbow'
    
    fig = px.bar(
        df_agg,
        x='zone',
        y='pm25_mean',
        color='pm25_mean',
        facet_col='country',
        facet_col_wrap=3,
        color_continuous_scale=color_scale,
        labels={'pm25_mean': 'Avg PM2.5 (µg/m³)', 'zone': 'Zone Type'},
        title=f'Pollution vs Zone Type vs Country<br><sub>Small Multiples Approach | Color: {"Sequential (Viridis)" if use_sequential else "Rainbow"}</sub>',
        height=500
    )
    
    # Add health threshold line
    fig.add_hline(y=35, line_dash="dash", line_color="#FF8A8A",  # Soft coral
                  annotation_text="Health Threshold", annotation_position="top right")
    
    fig.update_layout(
        template='plotly_white',
        font=dict(size=10)
    )
    
    return fig

def plot_heatmap_alternative(df):
    """Alternative visualization: Heatmap of Zone × Country."""
    # Create pivot table
    heatmap_data = df.groupby(['country', 'zone'])['pm25'].mean().reset_index()
    pivot = heatmap_data.pivot(index='zone', columns='country', values='pm25')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='Viridis',
        text=np.round(pivot.values, 1),
        texttemplate='%{text} µg/m³',
        textfont={"size": 11},
        colorbar=dict(title="PM2.5<br>µg/m³")
    ))
    
    fig.update_layout(
        title='Pollution Heatmap: Zone × Country<br><sub>Sequential color preserves perceptual uniformity</sub>',
        xaxis_title='Country',
        yaxis_title='Zone Type',
        height=400,
        template='plotly_white'
    )
    
    return fig

def plot_color_scale_comparison():
    """Demonstrate why sequential is better than rainbow."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            '✅ Sequential (Viridis): Equal steps in luminance',
            '❌ Rainbow: Unequal perception (yellow appears as peak)'
        ),
        vertical_spacing=0.15
    )
    
    values = np.linspace(0, 100, 100)
    
    # Sequential
    fig.add_trace(go.Heatmap(
        z=[values],
        colorscale='Viridis',
        showscale=False
    ), row=1, col=1)
    
    # Rainbow
    fig.add_trace(go.Heatmap(
        z=[values],
        colorscale='Rainbow',
        showscale=False
    ), row=2, col=1)
    
    fig.update_layout(
        height=300,
        title='Why Sequential Color Scales Are Better',
        template='plotly_white'
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    return fig