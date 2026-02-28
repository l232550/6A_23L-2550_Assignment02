#!/usr/bin/env python3
"""
Urban Environmental Intelligence Challenge - Main Pipeline
Run: python main.py
"""
import logging
from pathlib import Path
from fetching.opq_fetcher import OpenAQFetcher
from processing.preprocess import preprocess_raw
from modeling.pca_engine import PCAEngine
from modeling.temporal_engine import TemporalEngine
from modeling.distribution_engine import DistributionEngine  
from modeling.integrity_audit import IntegrityAudit
from visualization.pca_plots import plot_pca_clusters
from visualization.heatmap_plots import plot_temporal_analysis
from visualization.distribution_plots import plot_distributions
from visualization.regional_plots import plot_regional_integrity
from utils.io_utils import load_processed
from config import OPENAQ_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info(" Starting Urban Intelligence Pipeline")
    
    if OPENAQ_API_KEY == "your-api-key-here":
        logger.error("Set OPENAQ_API_KEY in config.py or export env var")
        return
    
    # 1. ETL (Big Data)
    logger.info(" Step 1: Data Fetching")
    fetcher = OpenAQFetcher(OPENAQ_API_KEY)
    df_raw = fetcher.run_full_etl()
    
    # 2. Preprocessing
    logger.info(" Step 2: Preprocessing")
    df_processed = preprocess_raw(df_raw)
    
    # 3. Task 1: PCA Dimensionality
    logger.info(" Step 3: PCA Analysis")
    pca_engine = PCAEngine(df_processed)
    X_pca, pca_model, scaler, loadings = pca_engine.fit_transform()
    
    # Task 1 viz (save PNG)
    zones = df_processed['zone'].iloc[:len(X_pca)].values
    fig_pca = plot_pca_clusters(X_pca, zones, loadings)
    fig_pca.write_image("reports/pca_clusters.png")
    
    # 4. Task 2: Temporal
    logger.info(" Step 4: Temporal Analysis")
    violations = TemporalEngine.violation_fraction(df_processed)
    hourly, monthly = TemporalEngine.periodic_signature(df_processed)
    fig_temporal = plot_temporal_analysis(violations, hourly)
    fig_temporal.write_image("reports/temporal_analysis.png")
    
    # 5. Task 3: Distributions
    logger.info(" Step 5: Distribution Analysis")
    ind_pm25 = df_processed[df_processed['zone'] == 'Industrial']['pm25']
    p99, extreme_prob = DistributionEngine.tail_analysis(ind_pm25)
    logger.info(f"99th percentile: {p99:.1f}, Extreme prob: {extreme_prob*100:.2f}%")
    fig_dist = plot_distributions(ind_pm25, p99)
    fig_dist.write_image("reports/distributions.png")
    
    # 6. Task 4: Integrity
    logger.info(" Step 6: Visual Audit")
    audit = IntegrityAudit.audit_3d_bar()
    logger.info(f"Audit: {audit}")
    df_for_regional = df_processed.sample(1000)
    fig_regional = plot_regional_integrity(df_for_regional)
    fig_regional.write_image("reports/regional_integrity.png")
    
    logger.info(" Pipeline COMPLETE! Check reports/ for PNGs, data/processed/ for data")
    logger.info("Run: streamlit run dashboard/app.py")

if __name__ == "__main__":
    main()
