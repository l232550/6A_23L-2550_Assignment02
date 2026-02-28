"""
task2_temporal.py  (temporal_engine.py)
-----------------
Task 2: High-Density Temporal Analysis

Problem: 100 line charts = unreadable clutter.
Solution: Heatmap (station × time) + hourly/monthly aggregations to reveal
          the "periodic signature" of PM2.5 health threshold violations.

Visualizations:
  1. Violation Heatmap     — station × month, color = violation rate
  2. Hourly Profile        — 24-hour PM2.5 cycle (daily pattern)
  3. Monthly Profile       — month-by-month PM2.5 trend (seasonal pattern)
  4. Horizon Chart         — compact 100-station time series overview
  5. Threshold Event Plot  — timeline of when stations breach 35 µg/m³
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import logging
import webbrowser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────

HEALTH_THRESHOLD = 35.0   # PM2.5 µg/m³ WHO/EPA daily threshold
OUTPUT_DIR       = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

ZONE_COLORS = {
           'Industrial':  "#F51A1A",  # Soft coral/light red
            'Residential': "#2DF02D",  # Soft mint/light green
            'Mixed':       "#494646"   # Soft light grey
}

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


# ── THEME ─────────────────────────────────────────────────────────────────────

def _base_layout(**extra) -> dict:
    d = dict(
        paper_bgcolor='#0f1117',
        plot_bgcolor='#1a1d27',
        font=dict(color='#e2e8f0', family='monospace'),
        hoverlabel=dict(bgcolor='#1e2433', bordercolor='#4a5568',
                        font=dict(size=12, family='monospace')),
        legend=dict(bgcolor='rgba(26,29,39,0.8)', bordercolor='#4a5568',
                    borderwidth=1, font=dict(size=11)),
    )
    d.update(extra)
    return d


def _axis_style(**extra) -> dict:
    d = dict(gridcolor='#2d3748', zerolinecolor='#4a5568', linecolor='#4a5568')
    d.update(extra)
    return d


def _axis_no_grid(**extra) -> dict:
    """Axis style with no gridlines — for cleaner charts."""
    d = dict(
        showgrid=False,
        zeroline=False,
        linecolor='#4a5568',
        tickcolor='#4a5568',
    )
    d.update(extra)
    return d


# ── DATA PREP ─────────────────────────────────────────────────────────────────

def prepare_temporal_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    elif 'datetimeUtc' in df.columns:
        df['timestamp'] = pd.to_datetime(df['datetimeUtc'], utc=True, errors='coerce')
    else:
        raise ValueError("No timestamp column found")

    df = df.dropna(subset=['timestamp'])

    df['hour']       = df['timestamp'].dt.hour
    df['month']      = df['timestamp'].dt.month
    df['month_name'] = df['timestamp'].dt.strftime('%b')
    df['date']       = df['timestamp'].dt.date
    df['date_str']   = df['timestamp'].dt.strftime('%Y-%m-%d')

    if 'pm25' in df.columns:
        df['pm25_daily_avg'] = df.groupby(
            ['location_id', 'date'])['pm25'].transform('mean')
        df['violation'] = (df['pm25_daily_avg'] > HEALTH_THRESHOLD).astype(int)
    else:
        logger.warning("'pm25' column not found — violation flags set to 0")
        df['pm25_daily_avg'] = 0.0
        df['violation']      = 0

    logger.info(f"Prepared temporal data: {len(df):,} rows | "
                f"{df['location_id'].nunique()} stations | "
                f"violation rate: {df['violation'].mean()*100:.1f}%")
    return df


# ── PLOT 1: VIOLATION HEATMAP ─────────────────────────────────────────────────

def fig_violation_heatmap(df: pd.DataFrame) -> go.Figure:
    if 'pm25' not in df.columns:
        logger.warning("No pm25 data for heatmap")
        return go.Figure()

    monthly = (
        df.groupby(['location_id', 'location_name', 'zone', 'month'])
        .agg(violation_rate=('violation', 'mean'),
             pm25_mean=('pm25', 'mean'))
        .reset_index()
    )
    monthly['violation_pct'] = monthly['violation_rate'] * 100

    pivot = monthly.pivot_table(
        index='location_id', columns='month',
        values='violation_pct', fill_value=0
    )

    name_map = df.drop_duplicates('location_id').set_index('location_id')
    zone_map  = name_map['zone'] if 'zone' in name_map.columns else {}

    labels = []
    for lid in pivot.index:
        name = name_map.loc[lid, 'location_name'] if lid in name_map.index else str(lid)
        zone = zone_map.get(lid, 'Mixed') if hasattr(zone_map, 'get') else 'Mixed'
        labels.append(f"{str(name)[:22]} [{zone[0]}]")

    row_totals    = pivot.sum(axis=1)
    sorted_idx    = row_totals.sort_values(ascending=False).index
    pivot_sorted  = pivot.loc[sorted_idx]
    labels_sorted = [labels[list(pivot.index).index(i)] for i in sorted_idx]

    for m in range(1, 13):
        if m not in pivot_sorted.columns:
            pivot_sorted[m] = 0
    pivot_sorted = pivot_sorted[[m for m in range(1, 13)]]

    z    = pivot_sorted.values
    text = [[f'{v:.1f}%' for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=MONTHS,
        y=labels_sorted,
        text=text,
        texttemplate='%{text}',
        textfont=dict(size=8),
        colorscale='YlOrRd',
        zmin=0, zmax=100,
        colorbar=dict(
            title=dict(text='Violation<br>Rate (%)', font=dict(size=11)),
            tickfont=dict(size=10),
            bgcolor='rgba(26,29,39,0.8)',
            bordercolor='#4a5568',
        ),
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Month: %{x}<br>'
            'Violation rate: %{z:.1f}%'
            '<extra></extra>'
        ),
    ))

    fig.update_layout(
        **_base_layout(height=max(500, len(labels_sorted) * 14 + 120)),
        title=dict(
            text=(f'<b>PM₂.₅ Health Threshold Violations by Station & Month</b><br>'
                  f'<sup>Color = % of days exceeding {HEALTH_THRESHOLD} µg/m³  '
                  f'| Sorted by annual violation rate  '
                  f'| [I]=Industrial [R]=Residential [M]=Mixed</sup>'),
            font=dict(size=14, color='#e2e8f0'),
            x=0.5, xanchor='center'
        ),
    )
    fig.update_xaxes(**_axis_style(title=dict(text='Month')))
    fig.update_yaxes(**_axis_style(
        title=dict(text='Monitoring Station'),
        tickfont=dict(size=9),
        showticklabels=False,
        automargin=True
    ))
    return fig


# ── PLOT 2: HOURLY PROFILE ────────────────────────────────────────────────────

def fig_hourly_profile(df: pd.DataFrame) -> go.Figure:
    if 'pm25' not in df.columns:
        return go.Figure()

    fig = go.Figure()

    hourly_all = df.groupby('hour')['pm25'].agg(['mean', 'std', 'median']).reset_index()

    fig.add_trace(go.Scatter(
        x=hourly_all['hour'],
        y=hourly_all['mean'],
        mode='lines',
        name='All stations (mean)',
        line=dict(color='#e2e8f0', width=3),
        hovertemplate='Hour %{x}:00<br>Mean PM₂.₅: %{y:.2f} µg/m³<extra>All</extra>',
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([hourly_all['hour'], hourly_all['hour'][::-1]]),
        y=pd.concat([hourly_all['mean'] + hourly_all['std'],
                     (hourly_all['mean'] - hourly_all['std'])[::-1]]),
        fill='toself',
        fillcolor='rgba(226,232,240,0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        name='±1 std dev',
        hoverinfo='skip',
    ))

    if 'zone' in df.columns:
        for zone in ['Industrial', 'Residential', 'Mixed']:
            subset = df[df['zone'] == zone]
            if subset.empty or 'pm25' not in subset.columns:
                continue
            hz = subset.groupby('hour')['pm25'].mean().reset_index()
            fig.add_trace(go.Scatter(
                x=hz['hour'], y=hz['pm25'],
                mode='lines+markers',
                name=zone,
                line=dict(color=ZONE_COLORS[zone], width=2, dash='dot'),
                marker=dict(size=5),
                hovertemplate=f'Hour %{{x}}:00<br>PM₂.₅: %{{y:.2f}} µg/m³<extra>{zone}</extra>',
            ))

    fig.add_hline(
        y=HEALTH_THRESHOLD,
        line=dict(color='#FF8A8A', width=1.5, dash='dash'),
        annotation_text=f'Health threshold ({HEALTH_THRESHOLD} µg/m³)',
        annotation_font=dict(color='#FF8A8A', size=11),
    )

    for x0, x1, label in [(7, 9, 'Morning rush'), (17, 19, 'Evening rush')]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor='rgba(255,215,0,0.08)',
                      line_width=0,
                      annotation_text=label,
                      annotation_font=dict(color='#FFD700', size=10),
                      annotation_position='top left')

    # ── FIX: xaxis/yaxis at top level, NOT inside title ──
    fig.update_layout(
        **_base_layout(height=460),
        title=dict(
            text=('<b>Daily Periodic Signature — 24-Hour PM₂.₅ Cycle</b><br>'
                  '<sup>Averaged across all stations | Reveals traffic-driven pollution peaks</sup>'),
            font=dict(size=14, color='#e2e8f0'),
            x=0.5, xanchor='center',
        ),
    )
    # Gridlines removed here — correct placement
    fig.update_xaxes(**_axis_no_grid(
        title=dict(text='Hour of Day'),
        tickmode='linear', tick0=0, dtick=2,
        tickvals=list(range(0, 24, 2)),
        ticktext=[f'{h:02d}:00' for h in range(0, 24, 2)],
    ))
    fig.update_yaxes(**_axis_no_grid(title=dict(text='PM₂.₅ (µg/m³)')))
    return fig


# ── PLOT 3: MONTHLY PROFILE ───────────────────────────────────────────────────

def fig_monthly_profile(df: pd.DataFrame) -> go.Figure:
    if 'pm25' not in df.columns:
        return go.Figure()

    monthly_all = df.groupby('month').agg(
        pm25_mean=('pm25', 'mean'),
        pm25_median=('pm25', 'median'),
        violation_count=('violation', 'sum'),
        total_readings=('pm25', 'count'),
    ).reset_index()
    monthly_all['violation_rate'] = (
        monthly_all['violation_count'] / monthly_all['total_readings'] * 100
    )
    monthly_all['month_name'] = monthly_all['month'].apply(lambda m: MONTHS[m-1])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    bar_colors = [
        '#FF8A8A' if v > 50 else '#FFB6C1' if v > 25 else '#B0E0E6'
        for v in monthly_all['violation_rate']
    ]
    fig.add_trace(go.Bar(
        x=monthly_all['month_name'],
        y=monthly_all['violation_rate'],
        name='Violation rate (%)',
        marker=dict(color=bar_colors, line=dict(color='white', width=0.4)),
        opacity=0.75,
        hovertemplate='<b>%{x}</b><br>Violation rate: %{y:.1f}%<extra></extra>',
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=monthly_all['month_name'],
        y=monthly_all['pm25_mean'],
        mode='lines+markers',
        name='Mean PM₂.₅',
        line=dict(color='#FFD700', width=2.5),
        marker=dict(size=8, color='#FFD700', line=dict(color='white', width=1.5)),
        hovertemplate='<b>%{x}</b><br>Mean PM₂.₅: %{y:.2f} µg/m³<extra></extra>',
    ), secondary_y=True)

    fig.add_hline(
        y=HEALTH_THRESHOLD,
        line=dict(color='#FF8A8A', width=1.5, dash='dash'),
        annotation_text=f'{HEALTH_THRESHOLD} µg/m³ threshold',
        annotation_font=dict(color='#FF8A8A', size=10),
        secondary_y=True
    )

    # ── FIX: xaxis/yaxis at top level, NOT inside title ──
    fig.update_layout(
        **_base_layout(height=440),
        title=dict(
            text=('<b>Monthly Periodic Signature — Seasonal PM₂.₅ Pattern</b><br>'
                  '<sup>Bars = % days above threshold | Line = mean PM₂.₅ | '
                  'Red bars > 50% violation rate</sup>'),
            font=dict(size=14, color='#e2e8f0'),
            x=0.5, xanchor='center',
        ),
        barmode='group',
    )
    # Gridlines removed here — correct placement
    fig.update_xaxes(**_axis_no_grid(title=dict(text='Month')))
    fig.update_yaxes(
        **_axis_no_grid(title=dict(text='Violation Rate (%)')),
        secondary_y=False, range=[0, 105]
    )
    fig.update_yaxes(
        **_axis_no_grid(title=dict(text='Mean PM₂.₅ (µg/m³)')),
        secondary_y=True
    )
    return fig


# ── PLOT 4: THRESHOLD EVENT TIMELINE ──────────────────────────────────────────

def fig_threshold_events(df: pd.DataFrame) -> go.Figure:
    if 'pm25' not in df.columns:
        return go.Figure()

    violations = (
        df[df['violation'] == 1]
        .drop_duplicates(['location_id', 'date_str'])
        [['location_id', 'location_name', 'zone', 'date_str', 'pm25_daily_avg']]
        .copy()
    )

    if violations.empty:
        fig = go.Figure()
        fig.add_annotation(
            text='No PM₂.₅ threshold violations found in dataset',
            xref='paper', yref='paper', x=0.5, y=0.5,
            font=dict(size=16, color='#94a3b8'), showarrow=False
        )
        fig.update_layout(**_base_layout(height=300))
        return fig

    violations['excess'] = violations['pm25_daily_avg'] - HEALTH_THRESHOLD
    violations['label']  = violations.apply(
        lambda r: f"{str(r['location_name'])[:22]} [{str(r.get('zone','?'))[0]}]",
        axis=1
    )

    freq_order = (
        violations.groupby('location_id')['date_str']
        .count().sort_values(ascending=True).index
    )
    label_order = [
        violations[violations['location_id'] == i]['label'].iloc[0]
        for i in freq_order if i in violations['location_id'].values
    ]

    fig = go.Figure()

    for zone in ['Industrial', 'Residential', 'Mixed']:
        subset = violations[violations['zone'] == zone]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset['date_str'],
            y=subset['label'],
            mode='markers',
            name=zone,
            marker=dict(
                color=ZONE_COLORS[zone],
                size=np.clip(subset['excess'] / 5 + 4, 4, 20),
                opacity=0.7,
                line=dict(color='white', width=0.3)
            ),
            hovertemplate=(
                '<b>%{y}</b><br>'
                'Date: %{x}<br>'
                'PM₂.₅: %{customdata[0]:.1f} µg/m³<br>'
                f'Excess: %{{customdata[1]:.1f}} µg/m³ above {HEALTH_THRESHOLD}'
                '<extra>' + zone + '</extra>'
            ),
            customdata=subset[['pm25_daily_avg', 'excess']].values,
        ))

    fig.add_vline(x='2025-01-01', line=dict(color='#4a5568', width=0.5))

    # ── FIX: xaxis/yaxis at top level, NOT inside title ──
    fig.update_layout(
        **_base_layout(height=max(500, len(label_order) * 16 + 150)),
        title=dict(
            text=('<b>Health Threshold Violation Events — 2025</b><br>'
                  f'<sup>Each dot = one station-day exceeding {HEALTH_THRESHOLD} µg/m³ | '
                  'Dot size = magnitude of excess | Color = zone type | '
                  'Hover any dot to see station name & exact values</sup>'),
            font=dict(size=14, color='#e2e8f0'),
            x=0.5, xanchor='center',
        ),
        margin=dict(l=20, r=40, t=80, b=60),  # tight left margin — no labels to make room for
    )
    fig.update_xaxes(**_axis_no_grid(
        title=dict(text='2025'),
        tickangle=0,
        tickfont=dict(size=11),
        tickmode='array',
        tickvals=[
            '2025-01-01', '2025-02-01', '2025-03-01', '2025-04-01',
            '2025-05-01', '2025-06-01', '2025-07-01', '2025-08-01',
            '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01',
        ],
        ticktext=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
        range=['2024-12-20', '2026-01-10'],
    ))
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        showline=False,        # ← removes the y-axis line itself, killing the gap
        ticks='',              # ← removes tick marks so no reserved space
        title=dict(text=''),   # ← empty title so no extra padding
        categoryorder='array',
        categoryarray=label_order,
        automargin=False,      # ← don't auto-expand margin for hidden labels
    )
    return fig


# ── TEXT SUMMARY ──────────────────────────────────────────────────────────────

def get_text_summary(df: pd.DataFrame) -> str:
    lines = [
        "=" * 55,
        "TASK 2 — TEMPORAL ANALYSIS RESULTS",
        "=" * 55,
        f"Total readings        : {len(df):,}",
        f"Stations              : {df['location_id'].nunique()}",
        f"Date range            : {df['date_str'].min()} → {df['date_str'].max()}",
        f"Health threshold      : {HEALTH_THRESHOLD} µg/m³ (PM₂.₅)",
        f"Overall violation rate: {df['violation'].mean()*100:.2f}%",
        "",
    ]

    if 'zone' in df.columns:
        lines.append("VIOLATION RATE BY ZONE:")
        for zone, grp in df.groupby('zone'):
            rate = grp['violation'].mean() * 100
            lines.append(f"  {zone:<15} {rate:.2f}%")

    lines += ["", "MONTHLY VIOLATION RATES:"]
    monthly = df.groupby('month')['violation'].mean() * 100
    for m, rate in monthly.items():
        lines.append(f"  {MONTHS[int(m)-1]:<5} {rate:.2f}%")

    lines += ["", "PEAK VIOLATION HOURS (top 5):"]
    if 'pm25' in df.columns:
        hourly = df.groupby('hour')['pm25'].mean().nlargest(5)
        for h, val in hourly.items():
            lines.append(f"  {int(h):02d}:00   {val:.2f} µg/m³")

    lines += ["", "TOP 10 MOST VIOLATING STATIONS:"]
    top_stations = (
        df.groupby(['location_id', 'location_name', 'zone'])['violation']
        .mean().mul(100).nlargest(10).reset_index()
    )
    for _, row in top_stations.iterrows():
        name = str(row.get('location_name', row['location_id']))[:30]
        zone = str(row.get('zone', '?'))
        lines.append(f"  {name:<32} {zone:<12} {row['violation']:.1f}%")

    lines += ["", "=" * 55]
    return '\n'.join(lines)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_task2(
    data_path: str = "data/processed/task2_temporal_dataset.parquet",
    save_html: bool = True,
    open_browser: bool = True,
):
    logger.info("=" * 55)
    logger.info("TASK 2: TEMPORAL ANALYSIS")
    logger.info("=" * 55)

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} rows | columns: {list(df.columns)}")

    if 'zone' not in df.columns:
        logger.warning("No 'zone' column — labelling all as Mixed")
        df['zone'] = 'Mixed'

    df = prepare_temporal_data(df)

    logger.info("Building figures...")
    figures = {
        'violation_heatmap': fig_violation_heatmap(df),
        'hourly_profile':    fig_hourly_profile(df),
        'monthly_profile':   fig_monthly_profile(df),
        'threshold_events':  fig_threshold_events(df),
    }

    if save_html:
        first_path = None
        for name, fig in figures.items():
            out = OUTPUT_DIR / f"task2_{name}.html"
            fig.write_html(str(out))
            logger.info(f"Saved: {out}")
            if first_path is None:
                first_path = out

        summary = get_text_summary(df)
        txt_path = OUTPUT_DIR / "task2_results.txt"
        txt_path.write_text(summary, encoding='utf-8')
        logger.info(f"Saved: {txt_path}")
        print(summary)

        if open_browser and first_path:
            webbrowser.open(str(first_path.resolve()))

    logger.info("✅ Task 2 complete")
    return df, figures


if __name__ == "__main__":
    run_task2(save_html=True, open_browser=True)