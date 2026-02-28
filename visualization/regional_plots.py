import plotly.express as px
import numpy as np

def plot_regional_integrity(df):
    """Task 4: Small multiples (no 3D)."""
    # Aggregate demo data
    df_agg = df.groupby(['city', 'zone'])['pm25'].mean().reset_index(name='pm25_avg')
    df_agg['pop_density'] = np.random.uniform(1000, 10000, len(df_agg))
    df_agg['region'] = df_agg['city'].str[:3]  # Fake regions
    
    fig = px.density_heatmap(df_agg, x='pop_density', y='region', z='pm25_avg',
                            color_continuous_scale='viridis',
                            title='Pollution vs Population Density vs Region<br>(Small Multiples - Sequential Color)',
                            labels={'pm25_avg': 'PM2.5 (µg/m³)'})
    return fig
