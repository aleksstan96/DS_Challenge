import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import nominal_features_registration, nominal_features_prev_lives

def check_values(dataset_paths, features):
    """
    Check for value mismatches across multiple datasets.
    
    Args:
        dataset_paths: List of paths to CSV datasets to compare
        features: List of features to check
    """
    # Load all datasets
    datasets = [pd.read_csv(path, index_col=0) for path in dataset_paths]
    
    mismatched_features = []
    for feature in features:
        print(f"\nChecking feature: {feature}")
        
        # Get unique values for each dataset
        all_value_sets = [set(df[feature].fillna("NaN").unique()) for df in datasets]
        
        # Get union of all unique values
        all_possible_values = set.union(*all_value_sets)
        
        # Check for mismatches
        has_mismatch = False
        for i, values in enumerate(all_value_sets):
            missing_values = all_possible_values - values
            if missing_values:
                if not has_mismatch:
                    print("❌ Values mismatch detected!")
                    has_mismatch = True
                print(f"Values missing in dataset {i} ({dataset_paths[i]}): {missing_values}")
        
        if has_mismatch:
            mismatched_features.append(feature)
        else:
            print("✓ Values match")
            print(f"Unique values: {sorted(all_possible_values)}")
    
    if mismatched_features:
        print(f"\nSummary: Found mismatches in {len(mismatched_features)} features:")
        for feature in mismatched_features:
            print(f"- {feature}")

def main():
    # check_values(['data/registration_data_training.csv', 'data/registration_data_test.csv'], nominal_features_registration)
    # check_values(['data/previous_lives_training_data.csv', 'data/previous_lives_test_data.csv'], nominal_features_prev_lives)
    # check_values(['data/registration_data_training.csv', 'data/registration_data_test.csv', 'data/previous_lives_training_data.csv', 'data/previous_lives_test_data.csv'], np.intersect1d(nominal_features_registration, nominal_features_prev_lives))

    # check cleaned data
    check_values(['cleaned_data/registration_data_training.csv', 'cleaned_data/registration_data_test.csv'], nominal_features_registration)
    check_values(['cleaned_data/previous_lives_training_data.csv', 'cleaned_data/previous_lives_test_data.csv'], nominal_features_prev_lives)
    check_values(['cleaned_data/registration_data_training.csv', 'cleaned_data/registration_data_test.csv', 'cleaned_data/previous_lives_training_data.csv', 'cleaned_data/previous_lives_test_data.csv'], np.intersect1d(nominal_features_registration, nominal_features_prev_lives))

if __name__ == "__main__":
    main()  
