import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import logging

logger = logging.getLogger(__name__)

def assign_class_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign class labels based on 'WIND' speed:
    0 — Tropical System (TS): wind < 64 knots
    1 — Moderate Hurricane (MH): 64 <= wind <= 95 knots
    2 — Severe Hurricane (SH): wind >= 96 knots
    """
    logger.info("Assigning class labels...")
    conditions = [
        (df['WIND'] < 64),
        (df['WIND'] >= 64) & (df['WIND'] <= 95),
        (df['WIND'] >= 96)
    ]
    choices = [0, 1, 2]
    df['LABEL'] = np.select(conditions, choices, default=np.nan)
    return df

def stratified_group_split(df: pd.DataFrame, group_col='SID', test_size=0.2, random_state=42):
    """
    Perform an 80/20 train-test split ensuring no single storm ('SID') 
    appears in both the train and test sets.
    """
    logger.info(f"Performing {1-test_size}/{test_size} GroupShuffleSplit by {group_col}...")
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    train_idx, test_idx = next(gss.split(df, groups=df[group_col]))
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    logger.info(f"Train split size: {len(train_df)}")
    logger.info(f"Test split size: {len(test_df)}")
    
    return train_df, test_df

def balance_training_set(train_df: pd.DataFrame, random_state=42) -> pd.DataFrame:
    """
    Balance the training set via stratified undersampling to the size of the smallest class.
    """
    logger.info("Balancing training set via undersampling...")
    class_counts = train_df['LABEL'].value_counts()
    logger.info(f"Class distribution before balancing:\n{class_counts}")
    
    min_class_size = class_counts.min()
    logger.info(f"Undersampling to {min_class_size} samples per class.")
    
    balanced_dfs = []
    for cls in class_counts.index:
        cls_df = train_df[train_df['LABEL'] == cls]
        cls_sampled_df = cls_df.sample(n=min_class_size, random_state=random_state)
        balanced_dfs.append(cls_sampled_df)
        
    balanced_train_df = pd.concat(balanced_dfs).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    logger.info(f"Balanced training set size: {len(balanced_train_df)}")
    return balanced_train_df

def standardise_and_rescale(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list):
    """
    Standardisation (zero mean, unit var) using TRAINING SET statistics only.
    Then Rescale to [0, pi] linearly using TRAINING SET min/max.
    """
    logger.info("Standardising and rescaling features to [0, pi]...")
    
    X_train = train_df[features].values
    X_test = test_df[features].values
    
    # 1. Standardise (Z-score normalisation)
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    
    # Handle zero std division safely
    sigma[sigma == 0] = 1e-8
    
    X_train_std = (X_train - mu) / sigma
    X_test_std = (X_test - mu) / sigma
    
    # 2. Rescale to [0, pi]
    min_vals = np.min(X_train_std, axis=0)
    max_vals = np.max(X_train_std, axis=0)
    
    val_range = max_vals - min_vals
    val_range[val_range == 0] = 1e-8
    
    X_train_rescaled = np.pi * (X_train_std - min_vals) / val_range
    X_test_rescaled = np.pi * (X_test_std - min_vals) / val_range
    
    return X_train_rescaled, X_test_rescaled

def get_processed_data(df: pd.DataFrame, random_state=42):
    """
    Main preprocessing pipeline: Label -> Split -> Balance Train -> Scale both.
    Returns:
        X_train, y_train, X_test, y_test
    """
    df = assign_class_labels(df)
    train_df, test_df = stratified_group_split(df, group_col='SID', test_size=0.2, random_state=random_state)
    
    # The test set keeps its natural imbalanced distribution
    balanced_train_df = balance_training_set(train_df, random_state=random_state)
    
    features = ['WIND', 'PRES', 'LAT', 'LON', 'USA_RMW', 'STORM_SPEED']
    
    X_train_scaled, X_test_scaled = standardise_and_rescale(balanced_train_df, test_df, features)
    
    y_train = balanced_train_df['LABEL'].values.astype(int)
    y_test = test_df['LABEL'].values.astype(int)
    
    return X_train_scaled, y_train, X_test_scaled, y_test
