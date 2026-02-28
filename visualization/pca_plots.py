import plotly.express as px
import pandas as pd

def plot_pca_clusters(X_pca, zones, loadings):
    """Task 1 viz."""
    df_plot = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_plot['zone'] = zones
    
    fig = px.scatter(df_plot, x='PC1', y='PC2', color='zone',
                     title='PCA: Industrial vs Residential Clustering',
                     labels={'PC1': 'PC1 (Pollution Axis)', 'PC2': 'PC2 (Climate Axis)'})
    
    # Loadings annotation
    top_loadings = loadings.head(3).round(3).to_string()
    fig.add_annotation(text=f"Top Loadings PC1:<br>{top_loadings}", 
                       xref="paper", yref="paper", x=0.02, y=0.98, 
                       showarrow=False, bgcolor="white")
    
    return fig
