"""
task1_pca.py
------------
Task 1: Dimensionality Reduction via PCA

Pipeline:
  Load station-level data → StandardScaler → PCA(2) →
  Scatter plot colored by zone → Loadings biplot → Variance explained bar

Outputs:
  outputs/task1_pca_scatter.png    ← main colored scatter (Industrial vs Residential)
  outputs/task1_loadings.png       ← loadings biplot (what drives each axis)
  outputs/task1_variance.png       ← scree/variance explained
  outputs/task1_results.txt        ← numeric summary for report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging
import plotly.express as px
import plotly.graph_objects as go

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────

REQUIRED_PARAMS   = ['pm25', 'pm10', 'no2', 'o3', 'temperature', 'humidity']
PARAM_LABELS      = {          # clean labels for plots
    'pm25':        'PM₂.₅',
    'pm10':        'PM₁₀',
    'no2':         'NO₂',
    'o3':          'O₃',
    'temperature': 'Temperature',
    'humidity':    'Humidity',
}

# UPDATED: Fresh, light colors for zones
ZONE_STYLE = {
    'Industrial':  {'color': '#F51A1A', 'marker': 'o', 'zorder': 4, 'size': 120},  # Soft coral/light red
    'Residential': {'color': '#2DF02D', 'marker': 's', 'zorder': 3, 'size': 100},  # Soft mint/light green
    'Mixed':       {'color': '#494646', 'marker': '^', 'zorder': 2, 'size':  90},  # Soft light grey
}

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── ENGINE ────────────────────────────────────────────────────────────────────

class PCAEngine:
    """
    Fits PCA on station-level annual mean environmental variables.

    Why station-level means?
      Each station represents one monitoring location with a characteristic
      pollution profile. Using annual means reduces temporal noise and gives
      one clean data point per station — exactly what we want to compare
      Industrial vs Residential zones.

    Why PCA?
      PCA is the appropriate choice here because:
      - The 6 variables are correlated (e.g. pm25 & pm10 both rise in
        industrial areas), so PCA finds the true axes of variation
      - It's interpretable via loadings — we can explain WHAT drives separation
      - It preserves maximum variance in 2D, unlike random projections
    """

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.scaler       = StandardScaler()
        self.pca          = PCA(n_components=n_components, random_state=42)
        self.params        = None   # actual params found in data
        self.fitted        = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise → fit PCA → return df with PC1, PC2 columns added.

        Args:
            df: station-level dataframe with one row per station,
                must contain columns from REQUIRED_PARAMS + 'zone'

        Returns:
            df with added columns: PC1, PC2
        """
        # Use whichever required params are actually present
        self.params = [p for p in REQUIRED_PARAMS if p in df.columns]
        missing = set(REQUIRED_PARAMS) - set(self.params)
        if missing:
            logger.warning(f"Missing parameters: {missing} — PCA on {len(self.params)} vars")
        if len(self.params) < 2:
            raise ValueError(f"Need at least 2 parameters, found: {self.params}")

        # Drop any rows with NaN in parameter columns
        df_clean = df.dropna(subset=self.params).copy()
        dropped  = len(df) - len(df_clean)
        if dropped:
            logger.warning(f"Dropped {dropped} stations with missing parameter values")

        logger.info(f"Fitting PCA on {len(df_clean)} stations × {len(self.params)} parameters")

        # Step 1: Standardise
        # Critical: PCA is scale-sensitive. Without this, temperature (°C, ~20)
        # would dominate over NO₂ (µg/m³, ~50) purely due to numeric magnitude.
        X_scaled = self.scaler.fit_transform(df_clean[self.params])

        # Step 2: Fit PCA
        X_pca = self.pca.fit_transform(X_scaled)

        # Step 3: Attach PC scores to dataframe
        df_clean['PC1'] = X_pca[:, 0]
        df_clean['PC2'] = X_pca[:, 1]

        self.fitted    = True
        self.df_result = df_clean

        self._log_results()
        return df_clean

    def get_loadings(self) -> pd.DataFrame:
        """
        Return signed loadings DataFrame.
        Shape: (n_params × n_components)

        Loadings tell us: how much does each original variable contribute
        to each principal component axis?
        High |loading| = that variable strongly drives variance along that PC.
        Sign tells direction (positive/negative relationship).
        """
        if not self.fitted:
            raise RuntimeError("Call fit_transform() first")

        loadings = pd.DataFrame(
            self.pca.components_.T,           # shape: (n_params, n_components)
            index=[PARAM_LABELS.get(p, p) for p in self.params],
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )
        return loadings

    def get_variance_explained(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_ * 100

    def _log_results(self):
        var = self.get_variance_explained()
        logger.info(f"\nPCA Results:")
        logger.info(f"  Variance explained: PC1={var[0]:.1f}%  PC2={var[1]:.1f}%  "
                    f"Total={sum(var):.1f}%")
        loadings = self.get_loadings()
        logger.info(f"\nLoadings:\n{loadings.round(3)}")


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def plot_pca_scatter(df: pd.DataFrame, engine: PCAEngine, save_path: Path):
    """
    Main scatter plot: each point = one monitoring station.
    Color/shape = zone type. Axes = PC1, PC2.
    """
    var = engine.get_variance_explained()

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d27')

    # Plot each zone separately for legend
    for zone, style in ZONE_STYLE.items():
        mask = df['zone'] == zone
        subset = df[mask]
        if subset.empty:
            continue
        ax.scatter(
            subset['PC1'], subset['PC2'],
            c=style['color'],
            marker=style['marker'],
            s=style['size'],
            alpha=0.85,
            edgecolors='white',
            linewidths=0.5,
            zorder=style['zorder'],
            label=f"{zone} (n={len(subset)})"
        )

    # Annotate some notable stations if location_name is present
    if 'location_name' in df.columns:
        notable = df[df['zone'] == 'Industrial'].nlargest(3, 'PC1')
        for _, row in notable.iterrows():
            ax.annotate(
                row['location_name'][:20],
                (row['PC1'], row['PC2']),
                textcoords='offset points', xytext=(6, 4),
                fontsize=7, color='#FF8A8A', alpha=0.9
            )

    ax.set_xlabel(f'PC1 ({var[0]:.1f}% variance explained)',
                  color='white', fontsize=12)
    ax.set_ylabel(f'PC2 ({var[1]:.1f}% variance explained)',
                  color='white', fontsize=12)
    ax.set_title('PCA of Environmental Variables Across 100 Monitoring Stations\n'
                 'Colored by Urban Zone Classification',
                 color='white', fontsize=13, fontweight='bold', pad=15)

    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333344')

    ax.axhline(0, color='#333344', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='#333344', linewidth=0.8, linestyle='--')
    ax.grid(True, alpha=0.12, color='white')

    legend = ax.legend(
        fontsize=10, framealpha=0.3,
        facecolor='#1a1d27', edgecolor='#555566',
        labelcolor='white', markerscale=1.2
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_loadings_biplot(engine: PCAEngine, save_path: Path):
    """
    Loadings biplot: arrows show how each original variable maps onto PC space.
    Long arrow = variable strongly contributes to that component.
    Arrow direction = sign of contribution.
    """
    var      = engine.get_variance_explained()
    loadings = engine.get_loadings()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0f1117')

    # UPDATED: Fresh, light colors for variables
    colors = ['#FF8A8A', '#A8E6A8', '#FFD700', '#B0E0E6', '#E0E0E0', '#FFB6C1']

    # Left: Arrow biplot
    ax = axes[0]
    ax.set_facecolor('#1a1d27')

    for i, (param, row) in enumerate(loadings.iterrows()):
        ax.annotate(
            '', xy=(row['PC1'], row['PC2']), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=colors[i % len(colors)],
                            lw=2.2, mutation_scale=18)
        )
        offset_x = 0.04 if row['PC1'] >= 0 else -0.04
        offset_y = 0.04 if row['PC2'] >= 0 else -0.04
        ax.text(row['PC1'] + offset_x, row['PC2'] + offset_y,
                param, color=colors[i % len(colors)],
                fontsize=10, fontweight='bold', ha='center')

    ax.axhline(0, color='#333344', linewidth=0.8)
    ax.axvline(0, color='#333344', linewidth=0.8)
    ax.set_xlabel(f'PC1 ({var[0]:.1f}%)', color='white', fontsize=11)
    ax.set_ylabel(f'PC2 ({var[1]:.1f}%)', color='white', fontsize=11)
    ax.set_title('Loadings Biplot\n(Arrow length = contribution strength)',
                 color='white', fontsize=11, fontweight='bold')
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333344')
    ax.grid(True, alpha=0.12, color='white')

    # Right: Grouped bar chart of loadings per component
    ax2 = axes[1]
    ax2.set_facecolor('#1a1d27')

    x      = np.arange(len(loadings))
    width  = 0.35
    bars1  = ax2.bar(x - width/2, loadings['PC1'], width,
                     color='#FF8A8A', alpha=0.8, label=f'PC1 ({var[0]:.1f}%)')  # Soft coral
    bars2  = ax2.bar(x + width/2, loadings['PC2'], width,
                     color='#A8E6A8', alpha=0.8, label=f'PC2 ({var[1]:.1f}%)')  # Soft mint

    ax2.set_xticks(x)
    ax2.set_xticklabels(loadings.index, rotation=30, ha='right',
                        color='white', fontsize=9)
    ax2.set_ylabel('Loading Value (signed)', color='white', fontsize=11)
    ax2.set_title('PC Loadings per Variable\n(positive = same direction as PC axis)',
                  color='white', fontsize=11, fontweight='bold')
    ax2.axhline(0, color='white', linewidth=0.6, alpha=0.4)
    ax2.tick_params(colors='#aaaaaa')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333344')
    ax2.grid(True, alpha=0.12, color='white', axis='y')
    ax2.legend(fontsize=9, facecolor='#1a1d27', edgecolor='#555566',
               labelcolor='white', framealpha=0.4)

    plt.suptitle('PCA Loadings Analysis — Drivers of Environmental Variation',
                 color='white', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_variance_explained(engine: PCAEngine, save_path: Path):
    """Scree plot showing variance explained by each component."""
    # Fit full PCA to show all components
    scaler_full = StandardScaler()
    X_scaled    = scaler_full.fit_transform(engine.df_result[engine.params])
    pca_full    = PCA(n_components=len(engine.params))
    pca_full.fit(X_scaled)

    var_ratio  = pca_full.explained_variance_ratio_ * 100
    cumulative = np.cumsum(var_ratio)
    components = [f'PC{i+1}' for i in range(len(var_ratio))]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d27')

    # UPDATED: Fresh, light colors for bars
    bars = ax.bar(components, var_ratio, color='#B0E0E6', alpha=0.8, label='Individual')  # Light steel blue
    ax.plot(components, cumulative, 'o--', color='#FF8A8A',
            linewidth=2, markersize=7, label='Cumulative')  # Soft coral

    # Highlight PC1 and PC2 with fresh colors
    bars[0].set_color("#24BD1F")  # Soft coral for PC1
    bars[1].set_color("#1C1396")  # Soft mint for PC2

    # Annotate bars
    for i, (bar, val) in enumerate(zip(bars, var_ratio)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom',
                color='white', fontsize=9, fontweight='bold')

    # 80% threshold line
    ax.axhline(80, color='#FFD700', linewidth=1, linestyle=':', alpha=0.6)  # Gold
    ax.text(len(components) - 0.5, 81, '80% threshold',
            color='#FFD700', fontsize=8, ha='right')

    ax.set_xlabel('Principal Component', color='white', fontsize=11)
    ax.set_ylabel('Variance Explained (%)', color='white', fontsize=11)
    ax.set_title('Scree Plot — Variance Explained by Each Principal Component',
                 color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333344')
    ax.grid(True, alpha=0.12, color='white', axis='y')
    ax.legend(fontsize=10, facecolor='#1a1d27', edgecolor='#555566',
              labelcolor='white', framealpha=0.4)
    ax.set_ylim(0, max(cumulative) + 8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Saved: {save_path}")

# ── PLOTLY VISUALIZATIONS ─────────────────────────────────────────────────────

def plot_pca_scatter_plotly(df, engine, save_path="outputs/task1_pca_scatter.html"):
    """Interactive PCA scatter: PC1 vs PC2, colored/shaped by zone."""
    var = engine.get_variance_explained()
    
    # UPDATED: Fresh, light colors for zones
    fig = px.scatter(
        df, x='PC1', y='PC2', color='zone', symbol='zone',
        size_max=18, opacity=0.85,
        hover_data=['location_name'] if 'location_name' in df.columns else None,
        color_discrete_map={
            'Industrial':  "#F51A1A",  # Soft coral/light red
            'Residential': "#2DF02D",  # Soft mint/light green
            'Mixed':       "#494646"   # Soft light grey
        },
        symbol_map={
            'Industrial': 'circle',
            'Residential': 'square',
            'Mixed': 'triangle-up'
        }
    )
    
    fig.update_layout(
        title=f'PCA of Environmental Variables Across Stations<br>PC1 {var[0]:.1f}%, PC2 {var[1]:.1f}%',
        xaxis_title=f'PC1 ({var[0]:.1f}% variance explained)',
        yaxis_title=f'PC2 ({var[1]:.1f}% variance explained)',
        template='plotly_white'
    )
    
    # Show and save (optional)
    # fig.show()  # Comment this out if you don't want it to pop up
    fig.write_html(save_path)
    logger.info(f"Saved interactive scatter: {save_path}")
    
    # RETURN THE FIGURE!
    return fig


def plot_loadings_biplot_plotly(engine, save_path="outputs/task1_loadings.html"):
    """Interactive loadings biplot: arrows + bar chart of PC contributions."""
    loadings = engine.get_loadings()
    var = engine.get_variance_explained()
    
    fig = go.Figure()

    # UPDATED: Fresh, light colors for arrows
    arrow_colors = ["#E91515", "#3FE63F", '#FFD700', "#58DCEE", "#353131", "#916A70"]
    
    # Arrows for each variable
    for i, param in enumerate(loadings.index):
        color = arrow_colors[i % len(arrow_colors)]
        fig.add_trace(go.Scatter(
            x=[0, loadings.loc[param, 'PC1']],
            y=[0, loadings.loc[param, 'PC2']],
            mode='lines+text',
            line=dict(width=2.5, color=color),
            text=[None, param],
            textposition='top center',
            textfont=dict(color=color, size=12, family='DM Sans, sans-serif'),  # ← explicit color
            name=param,
            showlegend=True
        ))

    # Right: loadings as grouped bar chart with updated colors
    x = loadings.index
    fig.add_trace(go.Bar(
        x=x, y=loadings['PC1'], name=f'PC1 ({var[0]:.1f}%)',
        marker_color='#FF8A8A', opacity=0.8  # Soft coral
    ))
    fig.add_trace(go.Bar(
        x=x, y=loadings['PC2'], name=f'PC2 ({var[1]:.1f}%)',
        marker_color='#A8E6A8', opacity=0.8  # Soft mint
    ))

    fig.update_layout(
    title='Loadings Biplot — Drivers of Environmental Variation',
    xaxis=dict(
        title=dict(text=f'PC1 ({var[0]:.1f}%)', font=dict(color='#1A2E3B', size=13, family='DM Sans')),
        tickfont=dict(color='#607D8B'),
        gridcolor='rgba(0,0,0,0)',
        zeroline=True, zerolinecolor='#CBD5E0', zerolinewidth=1,
    ),
    yaxis=dict(
        title=dict(text=f'PC2 ({var[1]:.1f}%)', font=dict(color='#1A2E3B', size=13, family='DM Sans')),
        tickfont=dict(color='#607D8B'),
        gridcolor='rgba(0,0,0,0)',
        zeroline=True, zerolinecolor='#CBD5E0', zerolinewidth=1,
    ),
    template='plotly_white',
    font=dict(color='#1A2E3B', family='DM Sans, sans-serif'),
    paper_bgcolor='#FFFFFF',
    plot_bgcolor='#FFFFFF',
    )

    # fig.show()  # Comment this out
    fig.write_html(save_path)
    logger.info(f"Saved interactive loadings biplot: {save_path}")
    
    # RETURN THE FIGURE!
    return fig


def plot_variance_explained_plotly(engine, save_path="outputs/task1_variance.html"):
    """Scree plot with variance explained + cumulative curve."""
    # Full PCA for all components
    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(engine.df_result[engine.params])
    pca_full = PCA(n_components=len(engine.params))
    pca_full.fit(X_scaled)
    
    var_ratio = pca_full.explained_variance_ratio_ * 100
    cumulative = np.cumsum(var_ratio)
    components = [f'PC{i+1}' for i in range(len(var_ratio))]

    fig = go.Figure()

    # Individual variance bars with fresh colors
    fig.add_trace(go.Bar(
        x=components, y=var_ratio, name='Individual', 
        marker_color=['#FF8A8A', '#A8E6A8', '#B0E0E6', '#E0E0E0']  # Fresh colors
    ))

    # Cumulative variance line
    fig.add_trace(go.Scatter(
        x=components, y=cumulative, mode='lines+markers', name='Cumulative',
        line=dict(color='#FF8A8A', dash='dash')  # Soft coral
    ))

    # Highlight PC1 and PC2 with fresh colors
    fig.add_vrect(x0=-0.5, x1=0.5, fillcolor="#FF8A8A", opacity=0.2, line_width=0)  # Soft coral
    fig.add_vrect(x0=0.5, x1=1.5, fillcolor="#A8E6A8", opacity=0.2, line_width=0)  # Soft mint

    # 80% threshold line
    fig.add_hline(y=80, line_dash='dot', line_color='#FFD700', annotation_text='80% threshold',  # Gold
                  annotation_position='top right', annotation_font_color='#FFD700')

    fig.update_layout(
        title='Variance Explained by Each Principal Component',
        yaxis_title='Variance Explained (%)',
        template='plotly_white'
    )

    # fig.show()  # Comment this out
    fig.write_html(save_path)
    logger.info(f"Saved interactive variance plot: {save_path}")
    
    # RETURN THE FIGURE!
    return fig

# ── TEXT SUMMARY FOR REPORT ───────────────────────────────────────────────────

def save_text_summary(df: pd.DataFrame, engine: PCAEngine, save_path: Path):
    """Write numeric results to a text file for easy copy-paste into report."""
    var      = engine.get_variance_explained()
    loadings = engine.get_loadings()

    lines = [
        "=" * 60,
        "TASK 1: PCA RESULTS SUMMARY",
        "=" * 60,
        "",
        f"Stations analysed:      {len(df)}",
        f"Parameters used:        {', '.join(engine.params)}",
        "",
        "VARIANCE EXPLAINED",
        f"  PC1: {var[0]:.2f}%",
        f"  PC2: {var[1]:.2f}%",
        f"  Total (PC1+PC2): {sum(var):.2f}%",
        "",
        "LOADINGS (signed)",
        loadings.round(4).to_string(),
        "",
        "TOP DRIVERS OF PC1 (pollution axis):",
    ]

    top_pc1 = loadings['PC1'].abs().nlargest(3)
    for param, val in top_pc1.items():
        sign = "+" if loadings.loc[param, 'PC1'] > 0 else "-"
        lines.append(f"  {param}: {sign}{val:.4f}")

    lines += ["", "TOP DRIVERS OF PC2:"]
    top_pc2 = loadings['PC2'].abs().nlargest(3)
    for param, val in top_pc2.items():
        sign = "+" if loadings.loc[param, 'PC2'] > 0 else "-"
        lines.append(f"  {param}: {sign}{val:.4f}")

    if 'zone' in df.columns:
        lines += ["", "PC1 MEAN BY ZONE (higher = more 'industrial-like' on PC1):"]
        zone_pc1 = df.groupby('zone')['PC1'].mean().sort_values(ascending=False)
        for zone, val in zone_pc1.items():
            lines.append(f"  {zone}: {val:.4f}")

    lines += ["", "=" * 60]

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    # Also print to console
    print('\n'.join(lines))
    logger.info(f"Saved: {save_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_task1(data_path: str = "data/processed/task1_pca_dataset.parquet"):
    logger.info("=" * 60)
    logger.info("TASK 1: PCA ANALYSIS")
    logger.info("=" * 60)

    # Load PCA-ready dataset (station-level means, already has 'zone')
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} stations from {data_path}")
    logger.info(f"Columns: {list(df.columns)}")

    if 'zone' not in df.columns:
        logger.warning("'zone' column missing — all stations labelled Mixed")
        df['zone'] = 'Mixed'

    # Check which params are available (handle _scaled suffix from preprocess.py)
    available = [p for p in REQUIRED_PARAMS if p in df.columns]
    if not available:
        # Try unscaled columns (preprocess.py saves both raw and scaled)
        logger.info("Raw param columns not found, checking for scaled versions...")
        available = [p for p in REQUIRED_PARAMS
                     if f"{p}_scaled" in df.columns]
        if available:
            for p in available:
                df[p] = df[f"{p}_scaled"]

    logger.info(f"Parameters for PCA: {available}")
    logger.info(f"Zone distribution:\n{df['zone'].value_counts()}")

    # Fit PCA
    engine = PCAEngine(n_components=2)
    df_result = engine.fit_transform(df)

    # Generate all plots
    plot_pca_scatter_plotly(df_result, engine)
    plot_loadings_biplot_plotly(engine)
    plot_variance_explained_plotly(engine)

    logger.info(f"\n✅ Task 1 complete. Outputs in {OUTPUT_DIR}/")
    return df_result, engine


if __name__ == "__main__":
    run_task1()