import pandas as pd
import logging
import os
import argparse
import joblib
from pathlib import Path

# Import the feature engineering pipeline from the separate file
from feature_engineering import get_feature_engineering_pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_run.log'), # Log file for the main run
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_RAW_PATH = PROJECT_ROOT / 'data' / 'raw' / 'data.csv'
DEFAULT_PROCESSED_PATH = PROJECT_ROOT / 'data' / 'processed'
DEFAULT_OUTPUT_FEATURES = DEFAULT_PROCESSED_PATH / 'processed_features.csv'
DEFAULT_OUTPUT_PIPELINE = DEFAULT_PROCESSED_PATH / 'feature_pipeline.pkl'

class DataValidator:
    """Utility for validating and converting data."""
    @staticmethod
    def validate_columns(df, required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {df.columns.tolist()}")
        
    @staticmethod
    def convert_to_datetime(df, date_col):
        """Converts a column to datetime objects, handling errors."""
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            if df[date_col].isnull().any():
                logger.warning(f"Some dates in column '{date_col}' failed to parse and were set to NaT.")
        return df

def load_and_prepare_data(input_path=DEFAULT_RAW_PATH):
    """
    Loads raw data, performs initial cleaning, and separates features (X) and target (y).
    This function operates at the transaction level.
    """
    logger.info(f"Loading data from {input_path}")
    raw_data = pd.read_csv(input_path)
    raw_data.columns = raw_data.columns.str.strip() # Clean column names
    
    logger.info(f"Loaded {len(raw_data)} records")
    logger.info("Data columns: " + ", ".join(raw_data.columns))

    # Define all expected columns for validation
    required_cols = [
        'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 
        'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 
        'ProductCategory', 'ChannelId', 'Amount', 'Value', 
        'TransactionStartTime', 'PricingStrategy', 'FraudResult'
    ]
    DataValidator.validate_columns(raw_data, required_cols)
    
    raw_data = DataValidator.convert_to_datetime(raw_data, 'TransactionStartTime')

    # X_data contains all features at the transaction level
    X_data = raw_data.drop(columns=['FraudResult'])
    # y_data is the target variable at the transaction level
    y_data = raw_data['FraudResult'] 

    return X_data, y_data

def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Credit Risk Feature Engineering Pipeline')
    parser.add_argument('--input', type=str, default=DEFAULT_RAW_PATH,
                        help='Path to input raw data file')
    parser.add_argument('--output-features', type=str, default=DEFAULT_OUTPUT_FEATURES,
                        help='Path to save processed features')
    parser.add_argument('--output-pipeline', type=str, default=DEFAULT_OUTPUT_PIPELINE,
                        help='Path to save feature pipeline')
    return parser.parse_args()

def main():
    """Main function to run the data processing and feature engineering pipeline."""
    try:
        args = parse_arguments()
        
        # Create output directory if it doesn't exist
        os.makedirs(DEFAULT_PROCESSED_PATH, exist_ok=True)
        
        # --- Step 1: Load and Prepare Data ---
        X_data, y_data = load_and_prepare_data(args.input)
        
        # --- Step 2: Build Feature Engineering Pipeline ---
        logger.info("Building feature engineering pipeline...")
        pipeline = get_feature_engineering_pipeline()
        
        # --- Step 3: Transform Data ---
        logger.info("Transforming data...")
        # Fit and transform the pipeline. Both X_data and y_data are transaction-level.
        # The pipeline will add new columns based on customer-level aggregates.
        processed_data = pipeline.fit_transform(X_data, y_data)
        
        # Add original CustomerId and FraudResult back to the processed data.
        # The ColumnTransformer's 'remainder'='drop' will remove these if not explicitly
        # included in feature lists, so we re-add them for convenience.
        processed_data['CustomerId'] = X_data['CustomerId'].values
        processed_data['FraudResult'] = y_data.values

        # Reorder columns to put CustomerId and FraudResult at the beginning for clarity
        cols = ['CustomerId', 'FraudResult'] + [col for col in processed_data.columns if col not in ['CustomerId', 'FraudResult']]
        processed_data = processed_data[cols]


        logger.info(f"Processed features shape: {processed_data.shape}")
        logger.info("Sample processed data:")
        logger.info(processed_data.head(3))
        
        # --- Step 4: Save Processed Data and Pipeline ---
        logger.info(f"Saving processed data to {args.output_features}")
        processed_data.to_csv(args.output_features, index=False)
        
        logger.info(f"Saving pipeline to {args.output_pipeline}")
        joblib.dump(pipeline, args.output_pipeline)
        
        logger.info("Feature engineering pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in feature processing: {str(e)}", exc_info=True)
        logger.info("Please ensure all required columns are present in your input data and dependencies are compatible.")
        raise

if __name__ == "__main__":
    main()

