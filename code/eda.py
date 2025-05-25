import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings # Import warnings to manage the FutureWarning

# DESCRIPTIVE STATISTICS

# Suppress the FutureWarning from seaborn for now, as it's not the critical error
warnings.filterwarnings('ignore', category=FutureWarning)
# Also suppress the RuntimeWarning for overflow encountered in cast (from reduce_mem_usage)
warnings.filterwarnings('ignore', category=RuntimeWarning)


# --- Memory Optimization Function (from previous script, essential for large data) ---
def reduce_mem_usage(df, verbose=True):
    """
    Iterates through all columns of a dataframe and modifies the data type to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object: # Exclude object type columns (strings)
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                # Keep as int64 if it doesn't fit smaller types
            else: # float
                # We will be careful here and avoid float16 for potentially problematic columns,
                # though the direct cast to float32 before plotting is the primary fix.
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                # Keep as float64 if it doesn't fit smaller types
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage after optimization is: {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.2f}% reduction)')
    return df

# --- 1. Configuration ---
DATA_PATH = '/kaggle/input/drw-crypto-market-prediction/'
TARGET = 'label'
PUBLIC_FEATURES = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
PROPRIETARY_FEATURES = [f'X{i}' for i in range(1, 891)] # Corrected X feature naming

# --- 2. Load Data ---
print("Loading training data for EDA...")
train_df = pd.read_parquet(f'{DATA_PATH}train.parquet')
train_df = reduce_mem_usage(train_df) # Optimize memory immediately
print(f"\nTraining data loaded. Shape: {train_df.shape}")

# --- 4. Target Variable Analysis ---
print("\n--- Target Variable ('label') Analysis ---")
print(f"Descriptive statistics for '{TARGET}':")
print(train_df[TARGET].describe())

plt.figure(figsize=(12, 6))
# Cast to float32 before plotting to avoid float16 index error
sns.histplot(train_df[TARGET].astype(np.float32), bins=50, kde=True)
plt.title(f'Distribution of {TARGET}')
plt.xlabel(TARGET)
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Zoom in on the central part of the distribution to see detail
plt.figure(figsize=(12, 6))
# Cast to float32 before plotting
sns.histplot(train_df[TARGET].astype(np.float32), bins=100, kde=True, stat='density')
plt.xlim(train_df[TARGET].quantile(0.01), train_df[TARGET].quantile(0.99)) # Focus on 98% of data
plt.title(f'Distribution of {TARGET} (Zoomed-in)')
plt.xlabel(TARGET)
plt.ylabel('Density')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Time series plot of the target variable (sampling for readability)
plt.figure(figsize=(15, 7))
# Resample to hourly mean; ensure the result is float32 if needed, but pandas often handles this
train_df[TARGET].resample('H').mean().plot(alpha=0.8)
plt.title(f'Hourly Mean of {TARGET} Over Time')
plt.xlabel('Timestamp')
plt.ylabel(TARGET)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- 5. Public Features Analysis ---
print("\n--- Public Features Analysis ---")

# Correlation matrix for public features and target
print("\nCorrelation Matrix (Public Features and Target):")
# Ensure the columns for correlation are in a compatible float format (float32 is fine)
correlation_matrix_public = train_df[PUBLIC_FEATURES + [TARGET]].astype(np.float32).corr()
print(correlation_matrix_public)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_public, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix: Public Features vs. Target')
plt.show()

# Plotting distributions for a few key public features
plt.figure(figsize=(15, 10))
for i, feature in enumerate(['volume', 'bid_qty', 'ask_qty'], 1):
    plt.subplot(2, 2, i)
    # Cast to float32 before plotting
    sns.histplot(train_df[feature].dropna().astype(np.float32), bins=50, kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.yscale('log') # Log scale is often useful for skewed financial data
    plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Time series plots for a few public features (hourly mean)
plt.figure(figsize=(15, 10))
for i, feature in enumerate(['volume', 'bid_qty'], 1):
    plt.subplot(2, 1, i)
    # Resample to hourly mean; ensure the result is float32 if needed
    train_df[feature].resample('H').mean().plot(alpha=0.8)
    plt.title(f'Hourly Mean of {feature} Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel(feature)
    plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- 6. Proprietary Features (X_i) Initial Look ---
print("\n--- Proprietary Features (X_i) Initial Look ---")

# Describe a sample of X features
# Ensure these columns are float32 for describe too, if they were downcasted to float16
print("\nDescriptive statistics for a sample of X features (X1, X2, X100, X500, X890):")
print(train_df[['X1', 'X2', 'X100', 'X500', 'X890']].astype(np.float32).describe())

# Check for NaNs among X features (top 10 with most NaNs)
print("\nMissing values in X features (top 10):")
x_null_counts = train_df[PROPRIETARY_FEATURES].isnull().sum()
print(x_null_counts.sort_values(ascending=False).head(10))

# Check for features with zero variance (constant features)
# Use .copy() to avoid SettingWithCopyWarning if you modify this later
temp_x_df_for_variance = train_df[PROPRIETARY_FEATURES].copy()
zero_variance_x = temp_x_df_for_variance.loc[:, temp_x_df_for_variance.std() == 0]
if not zero_variance_x.empty:
    print(f"\n{len(zero_variance_x.columns)} X features have zero variance (constant values):")
    print(zero_variance_x.columns.tolist())
else:
    print("\nNo X features found with zero variance.")
del temp_x_df_for_variance # Clean up

# Correlation of X features with the target (top 10 and bottom 10)
print("\nCorrelation of X features with 'label':")
# Cast X features to float32 before calculating correlation with label
x_correlations = train_df[PROPRIETARY_FEATURES].astype(np.float32).corrwith(train_df[TARGET].astype(np.float32)).sort_values(ascending=False)
print("\nTop 10 most correlated X features:")
print(x_correlations.head(10))
print("\nBottom 10 least (most negatively) correlated X features:")
print(x_correlations.tail(10))

# Plot distributions for a few selected X features (e.g., top 3 correlated)
if not x_correlations.empty:
    top_correlated_x = x_correlations.head(3).index.tolist()
    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(top_correlated_x, 1):
        plt.subplot(1, 3, i)
        # Cast to float32 before plotting
        sns.histplot(train_df[feature].dropna().astype(np.float32), bins=50, kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

print("\nEDA complete. Further deep dives into specific features or time periods can be done based on these initial findings.")


# TIME SERIES

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure seaborn styles are applied
sns.set(style='whitegrid')

# --- 1. Target Variable ('label') Time Series ---
plt.figure(figsize=(14, 6))
train_df[TARGET].plot(label='Original', alpha=0.5)
train_df[TARGET].rolling(window='6H').mean().plot(label='6-Hour Rolling Mean', linewidth=2)
plt.title('Target Variable (label) Over Time')
plt.xlabel('Timestamp')
plt.ylabel('label')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- 2. Public Features Time Series ---
public_features_to_plot = ['volume', 'bid_qty', 'ask_qty']
for feature in public_features_to_plot:
    plt.figure(figsize=(14, 6))
    train_df[feature].plot(label='Original', alpha=0.5)
    train_df[feature].rolling(window='6H').mean().plot(label='6-Hour Rolling Mean', linewidth=2)
    plt.title(f'Public Feature: {feature} Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel(feature)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# --- 3. Top 3 Correlated X Features Over Time ---
top_x_features = x_correlations.head(3).index.tolist()
for feature in top_x_features:
    plt.figure(figsize=(14, 6))
    train_df[feature].plot(label='Original', alpha=0.5)
    train_df[feature].rolling(window='6H').mean().plot(label='6-Hour Rolling Mean', linewidth=2)
    plt.title(f'Proprietary Feature: {feature} Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel(feature)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

