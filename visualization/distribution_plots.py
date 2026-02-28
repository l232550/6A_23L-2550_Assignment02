import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_distributions(pm25_data, p99):
    """Task 3: Peaks vs Tails."""
    fig = make_subplots(1, 2, subplot_titles=('Peak Density (KDE)', 'Tail Integrity (ECDF)'))
    
    # KDE for peaks
    fig.add_trace(go.Histogram(x=pm25_data, histnorm='probability density',
                              nbinsx=50, name='PM2.5 KDE'), row=1, col=1)
    
    # ECDF for tails
    sorted_pm = np.sort(pm25_data.dropna())
    ecdf = np.arange(1, len(sorted_pm)+1) / len(sorted_pm)
    fig.add_trace(go.Scatter(x=sorted_pm, y=ecdf, name='ECDF'), row=1, col=2)
    
    fig.add_vline(x=200, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_hline(y=0.99, line_dash="dash", line_color="orange", row=1, col=2)
    
    fig.update_layout(height=500, title_text="Distribution Modeling")
    return fig
