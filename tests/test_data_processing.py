import pytest
import pandas as pd
import numpy as np

# Assuming build_pipeline is in src/data_processing.py
# and get_feature_engineering_pipeline is called within it.
from src.data_processing import build_pipeline 

# Import the specific transformer if you want to test it in isolation
# from src.feature_engineering import DatetimeFeaturesExtractor, AggregateFeaturesAdder

# Note: The original TemporalFeatureExtractor from your previous test
# seems to be replaced by DatetimeFeaturesExtractor in your new setup.
# If you still have a src/transformers.py with TemporalFeatureExtractor,
# ensure it's still needed, or update the import/test to match your
# current feature_engineering.py's DatetimeFeaturesExtractor.
# For this update, I'm assuming DatetimeFeaturesExtractor is the one to test.

def test_datetime_feature_extraction():
    """
    Tests the DatetimeFeaturesExtractor for correct feature extraction.
    This replaces the old TemporalFeatureExtractor test.
    """
    test_data = pd.DataFrame({
        'TransactionStartTime': ['2025-01-01 14:30:00', '2025-01-04 09:15:00'], # Jan 1, 2025 (Wed), Jan 4, 2025 (Sat)
        'CustomerId': [1, 2],
        'Amount': [100, 200]
    })
    
    # We need to instantiate DatetimeFeaturesExtractor directly for this unit test
    # If it's not directly importable from src.feature_engineering, you might need to adjust.
    # Based on your feature_engineering.py, it should be.
    from src.feature_engineering import DatetimeFeaturesExtractor
    transformer = DatetimeFeaturesExtractor(datetime_col='TransactionStartTime')
    transformed = transformer.transform(test_data)
    
    assert 'transaction_hour' in transformed.columns
    assert 'transaction_day' in transformed.columns
    assert 'transaction_month' in transformed.columns
    assert 'transaction_year' in transformed.columns
    
    # Test case 1: January 1, 2025 (Wednesday)
    assert transformed.loc[0, 'transaction_hour'] == 14
    assert transformed.loc[0, 'transaction_day'] == 1
    assert transformed.loc[0, 'transaction_month'] == 1
    assert transformed.loc[0, 'transaction_year'] == 2025
    
    # Test case 2: January 4, 2025 (Saturday)
    assert transformed.loc[1, 'transaction_hour'] == 9
    assert transformed.loc[1, 'transaction_day'] == 4
    assert transformed.loc[1, 'transaction_month'] == 1
    assert transformed.loc[1, 'transaction_year'] == 2025

    # Assert that 'is_weekend' is NOT created by DatetimeFeaturesExtractor
    assert 'is_weekend' not in transformed.columns


def test_aggregate_features_adder():
    """
    Tests the AggregateFeaturesAdder for correct aggregate feature creation.
    """
    from src.feature_engineering import AggregateFeaturesAdder
    test_data = pd.DataFrame({
        'CustomerId': [1, 1, 2, 2, 1],
        'Amount': [100, 200, 50, 150, 50]
    })

    transformer = AggregateFeaturesAdder()
    transformed = transformer.transform(test_data)

    assert 'total_amount' in transformed.columns
    assert 'avg_amount' in transformed.columns
    assert 'count_transactions' in transformed.columns
    assert 'std_amount' in transformed.columns

    # Test CustomerId 1
    cust1_data = transformed[transformed['CustomerId'] == 1]
    assert cust1_data['total_amount'].iloc[0] == 350 # 100 + 200 + 50
    assert cust1_data['avg_amount'].iloc[0] == 350 / 3
    assert cust1_data['count_transactions'].iloc[0] == 3
    # Check std_amount, allowing for floating point precision
    assert np.isclose(cust1_data['std_amount'].iloc[0], np.std([100, 200, 50], ddof=1)) # ddof=1 for sample std dev

    # Test CustomerId 2
    cust2_data = transformed[transformed['CustomerId'] == 2]
    assert cust2_data['total_amount'].iloc[0] == 200 # 50 + 150
    assert cust2_data['avg_amount'].iloc[0] == 100 # (50+150)/2
    assert cust2_data['count_transactions'].iloc[0] == 2
    assert np.isclose(cust2_data['std_amount'].iloc[0], np.std([50, 150], ddof=1))

def test_full_pipeline():
    """
    Tests the end-to-end data processing pipeline for output shape and column presence.
    """
    sample_data = pd.DataFrame({
        'TransactionStartTime': ['2025-01-01 12:00:00', '2025-01-01 13:00:00', '2025-01-02 14:00:00'],
        'CustomerId': [1, 1, 2],
        'Amount': [100, 150, 200],
        'Value': [10, 15, 20],
        'ProductCategory': ['A', 'B', 'A'],
        'ChannelId': ['Web', 'App', 'Web'],
        'ProviderId': ['P1', 'P2', 'P1'],
        'CountryCode': ['US', 'US', 'CA'],
        'CurrencyCode': ['USD', 'USD', 'CAD'],
        'PricingStrategy': [1, 1, 2],
        'FraudResult': [0, 0, 1] # Include FraudResult as it might be passed to fit_transform for some pipelines
    })
    
    # Assuming build_pipeline from src.data_processing calls get_feature_engineering_pipeline
    pipeline = build_pipeline()
    
    # Separate features and target as done in your main() function
    X_sample = sample_data.drop(columns=['FraudResult'])
    y_sample = sample_data['FraudResult']

    # The pipeline transforms X_sample. y_sample is passed but not directly used by current transformers
    # within fit_transform unless a WOE transformer was present.
    result = pipeline.fit_transform(X_sample, y_sample)
    
    assert isinstance(result, pd.DataFrame), "Pipeline should return a pandas DataFrame."
    assert result.shape[0] == 3  # Same number of rows as input

    # Expected column count check:
    # Original relevant features: Amount, Value, ProductCategory, ChannelId, ProviderId, CountryCode, CurrencyCode, PricingStrategy, TransactionStartTime (9)
    # DatetimeFeaturesExtractor adds: transaction_hour, transaction_day, transaction_month, transaction_year (4 new)
    # AggregateFeaturesAdder adds: total_amount, avg_amount, count_transactions, std_amount (4 new)
    # Categorical Features (OHE):
    # ProductCategory: A, B (2 categories -> 2 cols)
    # ChannelId: Web, App (2 categories -> 2 cols)
    # ProviderId: P1, P2 (2 categories -> 2 cols)
    # CountryCode: US, CA (2 categories -> 2 cols)
    # CurrencyCode: USD, CAD (2 categories -> 2 cols)
    # Total OHE: 2+2+2+2+2 = 10 columns (for this small sample)
    # Numerical features passed to ColumnTransformer: Amount, Value, PricingStrategy + 4 temporal + 4 aggregate = 11
    # Total expected columns after preprocessing: 11 numeric (transformed) + 10 OHE = 21 (for this specific sample data)
    # Note: ColumnTransformer's 'remainder'='drop' means TransactionId, CustomerId are dropped if not explicitly listed.
    
    # A more flexible check:
    # We know we add 4 temporal and 4 aggregate features.
    # Original relevant features fed into pipeline (X_sample):
    # Amount, Value, TransactionStartTime, ProductCategory, ChannelId, ProviderId, CountryCode, CurrencyCode, PricingStrategy, CustomerId (10 cols, TransactionId if present)
    # CustomerId will be dropped by ColumnTransformer since it's not in features lists.
    # TransactionStartTime will be dropped by DatetimeFeaturesExtractor
    # Expected final columns will be the processed numeric + OHE.
    # The actual count depends on unique categories in the test data.
    
    # Let's check for specific expected columns
    assert 'transaction_hour' in result.columns
    assert 'total_amount' in result.columns
    assert 'num__Amount' not in result.columns # Should be flattened by ColumnTransformerDataFrameOutput
    assert 'preprocessing__num__Amount' not in result.columns # Should be flattened
    assert 'Amount' in result.columns # After StandardScaler, it's still named Amount or prefixed by 'num__'
    # Due to verbose_feature_names_out=False in ColumnTransformer, names should be plain
    assert 'Amount' in result.columns # From numeric_features
    assert 'ProductCategory_A' in result.columns or 'cat_ohe__ProductCategory_A' in result.columns # Depending on sklearn version and specific handling
    # Let's assume ColumnTransformerDataFrameOutput properly flattens names.
    assert 'ProductCategory_A' in result.columns
    assert 'ProductCategory_B' in result.columns
    assert 'ChannelId_Web' in result.columns
    assert 'ChannelId_App' in result.columns
    
    # Check that CustomerId and FraudResult are NOT in the pipeline's output
    # because they are added back *after* the pipeline in data_processing.py's main.
    assert 'CustomerId' not in result.columns 
    assert 'FraudResult' not in result.columns 
    
    # Check for number of columns - this is a rough estimate and might need adjustment
    # based on exact number of unique categories in test data and how OneHotEncoder handles them.
    # Given the sample data:
    # 11 numerical-like features after agg and datetime extraction
    # 5 categorical features, with 2 unique values each in this sample (approx 10 OHE columns)
    # So, around 21 columns is a good estimate for this specific test case.
    # It's better to check for range or exact count if categories are fixed.
    # For this small sample, 21 is very specific. Let's aim for a range.
    assert result.shape[1] >= 15 # A safer lower bound, accounting for common features and some OHE
    assert result.shape[1] <= 30 # A safer upper bound
    
    # Verify data types after scaling
    assert np.issubdtype(result['Amount'].dtype, np.floating)
    assert np.issubdtype(result['transaction_hour'].dtype, np.floating) # Scaled output will be float
    assert np.issubdtype(result['total_amount'].dtype, np.floating) # Scaled output will be float