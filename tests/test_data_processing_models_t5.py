# tests/test_data_processing_models_t5.py
import pytest
import pandas as pd
import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import load_data, preprocess_data

def test_load_data(tmp_path, monkeypatch):
    """Test that load_data correctly validates file existence"""
    # 1. First test successful load with mock file
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    
    test_data = pd.DataFrame({
        "CustomerId": [1, 2],
        "is_high_risk": [0, 1],
        "feature1": [0.5, 0.7]
    })
    test_data.to_csv(processed_dir / "data_with_target.csv", index=False)
    
    # Temporarily change the working directory
    monkeypatch.chdir(tmp_path)
    
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert set(df.columns) == {"CustomerId", "is_high_risk", "feature1"}

    # 2. Now test FileNotFoundError
    # Create a different directory where the file doesn't exist
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.chdir(empty_dir)
    
    with pytest.raises(FileNotFoundError) as excinfo:
        load_data()
    assert "Processed data not found at" in str(excinfo.value)
    assert "Run first: python src/target_engineering_t4.py" in str(excinfo.value)

def test_preprocess_data():
    """Test that preprocessing correctly splits data and validates columns"""
    # Create test data with more samples to avoid stratification issues
    test_df = pd.DataFrame({
        "CustomerId": [1, 2, 3, 4, 5, 6, 7, 8],
        "is_high_risk": [0, 1, 0, 1, 0, 1, 0, 1],  # Balanced classes
        "feature1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "feature2": [1, 2, 3, 4, 5, 6, 7, 8]
    })
    
    # Test correct splitting
    X_train, X_test, y_train, y_test = preprocess_data(test_df)
    assert len(X_train) == 6  # 75% of 8 (default test_size=0.25)
    assert len(X_test) == 2
    assert "is_high_risk" not in X_train.columns
    assert "CustomerId" not in X_train.columns
    assert set(X_train.columns) == {"feature1", "feature2"}
    assert len(y_train) == 6
    assert len(y_test) == 2
    
    # Test missing required columns
    with pytest.raises(ValueError, match="Missing required columns: {'is_high_risk'}"):
        preprocess_data(test_df.drop(columns=["is_high_risk"]))
        
    with pytest.raises(ValueError, match="Missing required columns: {'CustomerId'}"):
        preprocess_data(test_df.drop(columns=["CustomerId"]))