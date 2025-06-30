# src/target_engineering.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def calculate_rfm_features(transactions_df, snapshot_date=None):
    """
    Calculate RFM metrics from transaction data with proper datetime handling.
    """
    logger.info("Calculating RFM features")
    
    # Ensure datetime conversion
    transactions_df = transactions_df.copy()
    transactions_df['TransactionStartTime'] = pd.to_datetime(transactions_df['TransactionStartTime'])
    
    if snapshot_date is None:
        snapshot_date = transactions_df['TransactionStartTime'].max()
    elif isinstance(snapshot_date, str):
        snapshot_date = pd.to_datetime(snapshot_date)
    
    rfm = transactions_df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'CustomerId': 'count',
        'Amount': ['sum', 'mean']
    })
    
    rfm.columns = ['recency', 'frequency', 'monetary_total', 'monetary_mean']
    return rfm.reset_index()

def create_high_risk_label(rfm_df, n_clusters=3, random_state=42):
    """
    Create high-risk labels using KMeans clustering.
    """
    logger.info(f"Creating high-risk labels with {n_clusters} clusters")
    
    # Preprocess features
    scaler = StandardScaler()
    features = ['recency', 'frequency', 'monetary_total']
    X = scaler.fit_transform(rfm_df[features])
    
    # Cluster customers
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_df['cluster'] = kmeans.fit_predict(X)
    
    # Identify high-risk cluster
    cluster_stats = rfm_df.groupby('cluster')[features].mean()
    cluster_stats['score'] = (
        cluster_stats['recency'].rank(ascending=False) +
        cluster_stats['frequency'].rank(ascending=True) +
        cluster_stats['monetary_total'].rank(ascending=True)
    )
    
    high_risk_cluster = cluster_stats['score'].idxmax()
    rfm_df['is_high_risk'] = (rfm_df['cluster'] == high_risk_cluster).astype(int)
    
    logger.info(f"High-risk cluster: {high_risk_cluster}")
    logger.info(f"Risk distribution:\n{rfm_df['is_high_risk'].value_counts()}")
    
    return rfm_df[['CustomerId', 'is_high_risk']]

def add_target_variable(processed_df, transactions_df):
    """
    Full pipeline to add is_high_risk target variable.
    """
    # Calculate RFM features
    rfm_df = calculate_rfm_features(transactions_df)
    
    # Create high-risk labels
    risk_labels = create_high_risk_label(rfm_df)
    
    # Merge with processed data
    final_df = processed_df.merge(risk_labels, on='CustomerId', how='left')
    final_df['is_high_risk'] = final_df['is_high_risk'].fillna(0).astype(int)
    
    return final_df

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Load data with explicit datetime parsing
        transactions = pd.read_csv(
            "data/raw/data.csv",
            parse_dates=['TransactionStartTime']
        )
        processed_features = pd.read_csv("data/processed/processed_features.csv")
        
        # Process and save
        final_data = add_target_variable(processed_features, transactions)
        final_data.to_csv("data/processed/data_with_target.csv", index=False)
        logger.info("Successfully created target variable and saved data")
        
    except Exception as e:
        logger.error(f"Error in target engineering: {str(e)}", exc_info=True)
        raise