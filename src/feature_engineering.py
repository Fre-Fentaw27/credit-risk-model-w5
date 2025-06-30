import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, 
    StandardScaler, 
    OrdinalEncoder,
    MinMaxScaler 
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)

# --- Custom Transformers ---

class AggregateFeaturesAdder(BaseEstimator, TransformerMixin):
    """Adds aggregate transaction features per customer to the original DataFrame."""
    def __init__(self, group_col='CustomerId', agg_col='Amount'):
        self.group_col = group_col
        self.agg_col = agg_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info(f"Adding aggregate features for column '{self.agg_col}' grouped by '{self.group_col}'...")
        agg_df = X.groupby(self.group_col)[self.agg_col].agg(
            total_amount='sum',
            avg_amount='mean',
            count_transactions='count',
            std_amount='std'
        ).reset_index()
        
        X_transformed = X.copy()
        X_transformed = pd.merge(X_transformed, agg_df, on=self.group_col, how='left')
        
        X_transformed['std_amount'] = X_transformed['std_amount'].fillna(0)
        
        return X_transformed


class DatetimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    """Extracts hour, day, month, and year from TransactionStartTime."""
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info(f"Extracting datetime features from '{self.datetime_col}'...")
        X_transformed = X.copy()
        X_transformed[self.datetime_col] = pd.to_datetime(X_transformed[self.datetime_col], errors='coerce')
        
        X_transformed['transaction_hour'] = X_transformed[self.datetime_col].dt.hour
        X_transformed['transaction_day'] = X_transformed[self.datetime_col].dt.day
        X_transformed['transaction_month'] = X_transformed[self.datetime_col].dt.month
        X_transformed['transaction_year'] = X_transformed[self.datetime_col].dt.year
        
        return X_transformed

class ColumnTransformerDataFrameOutput(BaseEstimator, TransformerMixin):
    """Wraps ColumnTransformer to ensure DataFrame output with correct column names."""
    def __init__(self, column_transformer):
        self.column_transformer = column_transformer
        self.feature_names_out = None

    def fit(self, X, y=None):
        self.column_transformer.fit(X, y)
        self.feature_names_out = self.column_transformer.get_feature_names_out()
        return self

    def transform(self, X):
        transformed_data = self.column_transformer.transform(X)
        if hasattr(transformed_data, "toarray"):
            transformed_data = transformed_data.toarray()
        
        # Ensure the index is preserved for alignment with the target variable later
        return pd.DataFrame(transformed_data, columns=self.feature_names_out, index=X.index)


def get_feature_engineering_pipeline():
    """
    Builds and returns the full feature engineering pipeline.
    This pipeline operates on transaction-level data and adds/transforms features.
    """
    numeric_features = [
        "Amount", "Value", "PricingStrategy", 
        "total_amount", "avg_amount", "count_transactions", "std_amount", 
        "transaction_hour", "transaction_day", "transaction_month", "transaction_year"
    ]
    
    categorical_features_ohe = ["ProductCategory", "ChannelId", "ProviderId", "CountryCode", "CurrencyCode"] 
    
    # Define ordinal features if applicable, otherwise leave empty.
    # If a feature is listed here, it should NOT be in categorical_features_ohe.
    ordinal_features = [] 

    # --- Numeric processing pipeline ---
    numeric_transformer = Pipeline(steps=[
        ("imputer", KNNImputer(n_neighbors=5)), 
        ("scaler", StandardScaler())
    ])

    # --- One-Hot Categorical processing pipeline ---
    categorical_ohe_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)) 
    ])

    # --- Ordinal encoding pipeline (if ordinal_features is not empty) ---
    ordinal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)) 
    ])

    # --- ColumnTransformer to combine all preprocessing ---
    transformers_list = [
        ("num", numeric_transformer, numeric_features),
        ("cat_ohe", categorical_ohe_transformer, categorical_features_ohe)
    ]
    
    if ordinal_features: 
        transformers_list.append(("ordinal", ordinal_transformer, ordinal_features))

    preprocessor = ColumnTransformer(
        transformers=transformers_list,
        remainder='drop', # Drop any columns not specified (e.g., TransactionId, original TransactionStartTime, CustomerId if not in features)
        verbose_feature_names_out=False 
    )

    # --- Final full pipeline ---
    full_pipeline = Pipeline(steps=[
        ("agg_features", AggregateFeaturesAdder()),
        ("datetime_features", DatetimeFeaturesExtractor()),
        ("preprocessing", ColumnTransformerDataFrameOutput(preprocessor)), 
    ])

    return full_pipeline

if __name__ == "__main__":
    # This block is for testing the feature_engineering.py module independently.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Dummy data for testing pipeline construction
    data = {
        'CustomerId': [1, 1, 2, 2, 3],
        'Amount': [100, 200, 50, 150, 300],
        'Value': [10, 20, 5, 15, 30],
        'TransactionStartTime': ['2023-01-01T10:00:00Z', '2023-01-01T11:00:00Z', '2023-01-02T12:00:00Z', '2023-01-02T13:00:00Z', '2023-01-03T14:00:00Z'],
        'ProductCategory': ['Electronics', 'Electronics', 'Books', 'Books', 'Food'],
        'ChannelId': ['Web', 'Mobile', 'Web', 'Mobile', 'Web'],
        'ProviderId': ['A', 'A', 'B', 'B', 'C'],
        'PricingStrategy': [1, 2, 1, 2, 1],
        'CountryCode': ['US', 'US', 'CA', 'CA', 'MX'],
        'CurrencyCode': ['USD', 'USD', 'CAD', 'CAD', 'MXN'],
        'FraudResult': [0, 0, 0, 1, 0] # Example target, though not used by feature pipeline's fit_transform
    }
    df_test = pd.DataFrame(data)

    pipeline = get_feature_engineering_pipeline()
    logger.info("Pipeline created successfully.")

    # To actually transform for a test:
    X_test = df_test.drop(columns=['FraudResult'])
    y_test = df_test['FraudResult'] # WOE would use this, but it's excluded for now
    
    # The pipeline's fit_transform will use X_test (transaction-level)
    processed_df_test = pipeline.fit_transform(X_test, y_test) 
    logger.info("Processed data head:\n" + str(processed_df_test.head()))
    logger.info(f"Processed data shape: {processed_df_test.shape}")

