import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.express as px
from scipy import stats

def plot_distributions(pm25_data, p99, zone_name="Industrial"):
    """Task 3: Optimized plots for peaks (histogram) and tails (log-scale KDE)."""
    clean_data = pm25_data.dropna()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'📊 Peak Detection (Histogram)<br><sub>Reveals common pollution levels</sub>',
            f'📈 Tail Integrity (Log-Scale KDE)<br><sub>Exposes rare extreme events</sub>'
        ),
        horizontal_spacing=0.12
    )
    
    # LEFT: Histogram for peaks (linear scale)
    fig.add_trace(go.Histogram(
        x=clean_data,
        nbinsx=60,
        name='Frequency',
        marker=dict(color='#B0E0E6', line=dict(color='white', width=1)),  # Light steel blue
        opacity=0.8
    ), row=1, col=1)
    
    fig.add_vline(x=35, line_dash="dash", line_color="#FF8A8A",  # Soft coral
                  annotation_text="Health Threshold (35 µg/m³)",
                  annotation_position="top", row=1, col=1)
    
    # RIGHT: KDE for tails (LOG scale to reveal extreme events)
    log_data = np.log10(clean_data[clean_data > 0] + 1)
    
    fig.add_trace(go.Histogram(
        x=clean_data,
        histnorm='probability density',
        nbinsx=100,
        name='KDE',
        marker=dict(color='#FF8A8A'),  # Soft coral
        opacity=0.7
    ), row=1, col=2)
    
    fig.add_vline(x=200, line_dash="dash", line_color="#FF6B6B",  # Slightly darker red for contrast
                  annotation_text="Extreme Hazard (200 µg/m³)",
                  annotation_position="top", row=1, col=2)
    
    fig.add_vline(x=p99, line_dash="dot", line_color="#A8E6A8",  # Soft mint
                  annotation_text=f"99th Percentile ({p99:.1f})",
                  annotation_position="bottom", row=1, col=2)
    
    fig.update_xaxes(title_text="PM2.5 (µg/m³)", row=1, col=1)
    fig.update_xaxes(title_text="PM2.5 (µg/m³) [Log Scale]", type="log", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Probability Density", row=1, col=2)
    
    fig.update_layout(
        height=500,
        title_text=f"Distribution Analysis: {zone_name} Zone<br><sub>Left: Shows peaks | Right: Shows long tail</sub>",
        showlegend=False,
        template='plotly_white'
    )
    return fig


def plot_zone_comparison(df):
    """Compare distributions across zones using violin plots."""
    fig = go.Figure()
    
    zones = df['zone'].unique()
    # UPDATED: Fresh, light colors for zones
    colors = {'Industrial': '#FF8A8A', 'Residential': '#A8E6A8', 'Mixed': '#E0E0E0'}
    
    for zone in zones:
        zone_data = df[df['zone'] == zone]['pm25'].dropna()
        fig.add_trace(go.Violin(
            y=zone_data,
            name=zone,
            box_visible=True,
            meanline_visible=True,
            marker=dict(color=colors.get(zone, 'gray'))
        ))
    
    fig.add_hline(y=35, line_dash="dash", line_color="#FF8A8A",  # Soft coral
                  annotation_text="Health Threshold")
    fig.add_hline(y=200, line_dash="dash", line_color="#FF6B6B",  # Slightly darker red
                  annotation_text="Extreme Hazard")
    
    fig.update_layout(
        title="PM2.5 Distribution Comparison Across Zones",
        yaxis_title="PM2.5 (µg/m³)",
        xaxis_title="Zone Type",
        height=500,
        template='plotly_white'
    )
    return fig