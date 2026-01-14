import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Optional, List, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_columns(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Verifies that the specified columns exist in the DataFrame.

    Args:
        df: Input DataFrame.
        columns: List of column names to check.

    Raises:
        ValueError: If any column is missing.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

def analyze_fan_out(
    df: pd.DataFrame, 
    group_col: str, 
    target_col: str, 
    threshold: int = 1
) -> pd.DataFrame:
    """
    Identifies potential Mule Rings by analyzing the number of unique target entities 
    (e.g., accounts) associated with a grouping entity (e.g., customer name).

    Args:
        df: Input DataFrame.
        group_col: Column to group by (e.g., 'customer_name').
        target_col: Column to count unique values of (e.g., 'customer_account_number').
        threshold: Minimum number of unique targets to flag.

    Returns:
        DataFrame containing groups exceeding the threshold.
    """
    check_columns(df, [group_col, target_col])
    logger.info(f"Analyzing Fan-Out: Grouping by '{group_col}', counting unique '{target_col}'")

    grouped = df.groupby(group_col)[target_col].nunique().reset_index()
    grouped.columns = [group_col, 'unique_count']
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.histplot(grouped['unique_count'], log_scale=True, bins=20)
    plt.title(f'Fan-Out Distribution: {target_col} per {group_col}')
    plt.xlabel('Unique Count (Log Scale)')
    plt.ylabel('Frequency')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

    potential_mules = grouped[grouped['unique_count'] > threshold].sort_values('unique_count', ascending=False)
    logger.info(f"Found {len(potential_mules)} entities exceeding threshold {threshold}")
    
    return potential_mules

def analyze_benford(df: pd.DataFrame, amount_col: str) -> None:
    """
    Compares the distribution of the first digit of transaction amounts against Benford's Law.
    Used to detect artificial structuring of amounts.

    Args:
        df: Input DataFrame.
        amount_col: Column containing transaction amounts.
    """
    check_columns(df, [amount_col])
    logger.info(f"Analyzing Benford's Law on '{amount_col}'")

    # Extract first digit, ensuring we ignore 0 or negative values if any for Benford analysis
    # Convert to string, strip whitespace/potential signs, take first char
    first_digits = df[amount_col].astype(str).str.lstrip('-').str[0]
    
    # Filter for digits 1-9
    first_digits = first_digits[first_digits.isin([str(d) for d in range(1, 10)])].astype(int)
    
    if first_digits.empty:
        logger.warning("No valid positive amounts found for Benford analysis.")
        return

    observed_counts = first_digits.value_counts().sort_index()
    total_count = observed_counts.sum()
    observed_freq = observed_counts / total_count

    # Expected Benford Frequencies
    digits = np.arange(1, 10)
    expected_freq = np.log10(1 + 1/digits)

    # Plotting
    plt.figure(figsize=(10, 6))
    width = 0.35
    plt.bar(digits - width/2, observed_freq, width=width, label='Observed', alpha=0.7)
    plt.bar(digits + width/2, expected_freq, width=width, label='Expected (Benford)', alpha=0.7)
    plt.xticks(digits)
    plt.title("Benford's Law Analysis: Observed vs Expected First Digit Frequencies")
    plt.xlabel('First Digit')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.show()

def analyze_round_numbers(df: pd.DataFrame, amount_col: str) -> None:
    """
    Calculates and prints the percentage of transactions that are round numbers (integers).
    High ratios may indicate non-organic/bot behavior.

    Args:
        df: Input DataFrame.
        amount_col: Column containing transaction amounts.
    """
    check_columns(df, [amount_col])
    logger.info(f"Analyzing Round Numbers on '{amount_col}'")

    # Check if value is close to an integer
    is_round = (df[amount_col] % 1 == 0)
    round_count = is_round.sum()
    total_count = len(df)
    
    ratio = round_count / total_count if total_count > 0 else 0
    
    print(f"\n--- Round Number Analysis ---")
    print(f"Total Transactions: {total_count}")
    print(f"Round Number Transactions: {round_count}")
    print(f"Ratio: {ratio:.4f} ({ratio*100:.2f}%)")
    
    if ratio > 0.1: # Arbitrary threshold for warning
        print("Warning: High percentage of round numbers detected.")

def analyze_pass_through(
    df: pd.DataFrame, 
    id_col: str, 
    amount_col: str, 
    flow_col: str
) -> None:
    """
    Analyzes the Flow Ratio to detect Pass-Through/Mule accounts.
    Flow Ratio = abs(IN - OUT) / (IN + OUT).
    Ratio ~ 0.0 indicates Balanced Flow (Pass-Through).
    Ratio ~ 1.0 indicates Source/Sink.

    Args:
        df: Input DataFrame.
        id_col: Account ID column.
        amount_col: Transaction amount column.
        flow_col: Direction column (values should contain 'IN', 'OUT').
    """
    check_columns(df, [id_col, amount_col, flow_col])
    logger.info(f"Analyzing Pass-Through/Flow Ratio on '{id_col}'")

    # Normalize flow column to upper case just in case
    df = df.copy()
    df[flow_col] = df[flow_col].astype(str).str.upper()

    pivot = df.pivot_table(
        index=id_col, 
        columns=flow_col, 
        values=amount_col, 
        aggfunc='sum',
        fill_value=0
    )

    if 'IN' not in pivot.columns or 'OUT' not in pivot.columns:
        logger.warning("Data does not contain both 'IN' and 'OUT' flows necessary for this analysis.")
        return

    pivot['total_flow'] = pivot['IN'] + pivot['OUT']
    pivot['flow_ratio'] = abs(pivot['IN'] - pivot['OUT']) / pivot['total_flow']
    
    # Filter out inactive or zero-flow accounts if any
    pivot = pivot[pivot['total_flow'] > 0]

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.histplot(pivot['flow_ratio'], bins=20, kde=True)
    plt.title('Flow Ratio Distribution')
    plt.xlabel('Flow Ratio (|IN - OUT| / (IN + OUT))')
    plt.ylabel('Count of Accounts')
    plt.axvline(0.1, color='r', linestyle='--', label='Potential Mule Zone (<0.1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def analyze_sequence_length(df: pd.DataFrame, id_col: str, date_col: str) -> None:
    """
    Analyzes the number of active days per account to suggest Graph vs Sequence models.

    Args:
        df: Input DataFrame.
        id_col: Account ID column.
        date_col: Date column (must be datetime compatible).
    """
    check_columns(df, [id_col, date_col])
    logger.info(f"Analyzing Sequence Length (Active Days) on '{id_col}'")

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            logger.error(f"Could not convert '{date_col}' to datetime: {e}")
            return

    active_days = df.groupby(id_col)[date_col].apply(lambda x: x.dt.date.nunique())
    avg_days = active_days.mean()

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.histplot(active_days, bins=30)
    plt.title('Distribution of Active Days per Account')
    plt.xlabel('Number of Unique Active Days')
    plt.ylabel('Count of Accounts')
    plt.axvline(avg_days, color='k', linestyle='--', label=f'Mean: {avg_days:.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"\n--- Sequence Length Recommendation ---")
    print(f"Average Active Days per Account: {avg_days:.2f}")
    if avg_days < 3:
        print("Recommendation: Data is sparse. Suggest Graph Neural Networks (GNNs) or Static Features.")
    elif avg_days > 10:
        print("Recommendation: Long sequences detected. Suggest RNNs (LSTM/GRU) or Transformers.")
    else:
        print("Recommendation: Hybrid approach (GNN + Temporal features).")

def analyze_name_entropy(df: pd.DataFrame, name_col: str, amount_col: str) -> None:
    """
    Calculates Shannon Entropy of transaction amounts for frequent names.
    Detects clusters of "High Frequency + Low Entropy" (scripted bots).

    Args:
        df: Input DataFrame.
        name_col: Customer name column.
        amount_col: Transaction amount column.
    """
    check_columns(df, [name_col, amount_col])
    logger.info(f"Analyzing Name Entropy on '{name_col}'")

    # Filter for names with > 5 transactions
    name_counts = df[name_col].value_counts()
    frequent_names = name_counts[name_counts > 5].index
    
    if len(frequent_names) == 0:
        logger.warning("No names found with > 5 transactions.")
        return

    sub_df = df[df[name_col].isin(frequent_names)]

    def calc_entropy(series):
        # Calculate entropy of value counts (discrete distribution of amounts)
        probs = series.value_counts(normalize=True)
        return stats.entropy(probs)

    entropy_df = sub_df.groupby(name_col)[amount_col].apply(calc_entropy).reset_index()
    entropy_df.columns = [name_col, 'amount_entropy']
    entropy_df['transaction_count'] = entropy_df[name_col].map(name_counts)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=entropy_df, x='transaction_count', y='amount_entropy', alpha=0.6)
    plt.title('Name Frequency vs. Amount Entropy')
    plt.xlabel('Transaction Frequency')
    plt.ylabel('Amount Entropy (Shannon)')
    
    # Highlight potential bots (High Freq, Low Entropy)
    # Define thresholds for highlighting (e.g., top 50% freq, bottom 25% entropy)
    high_freq_thresh = entropy_df['transaction_count'].quantile(0.5)
    low_entropy_thresh = entropy_df['amount_entropy'].quantile(0.25)
    
    plt.axvline(high_freq_thresh, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(low_entropy_thresh, color='gray', linestyle=':', alpha=0.5)
    
    plt.text(high_freq_thresh * 1.1, low_entropy_thresh * 0.5, 
             "Potential Scripted/Bot Zone\n(High Freq, Low Entropy)", 
             color='red', fontsize=9)

    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    print("Generating dummy data for demonstration..." )
    
    # 1. Create Dummy Data
    np.random.seed(42)
    n_rows = 1000
    
    data = {
        'trxn_id': range(1, n_rows + 1),
        'amount': np.concatenate([
            np.random.exponential(scale=100, size=800), # Organic-ish
            np.array([500.0] * 50), # Structuring/Bot (Low entropy)
            np.random.uniform(10, 20, size=150) # Small values
        ]),
        'transaction_date': pd.date_range(start='2024-01-01', periods=n_rows, freq='h'),
        'cashflow_direction': np.random.choice(['IN', 'OUT'], size=n_rows),
        'customer_account_number': np.random.randint(1000, 1100, size=n_rows), # 100 accounts
        'customer_name': np.random.choice([f'User_{i}' for i in range(50)], size=n_rows),
        'customer_country': ['US'] * n_rows,
        'customer_city': ['New York'] * n_rows,
        'customer_bank_name': ['Bank A'] * n_rows
    }
    
    # Introduce some "Round Numbers"
    data['amount'][0:100] = np.round(data['amount'][0:100])
    
    df_dummy = pd.DataFrame(data)
    
    # Introduce a "Mule Ring" scenario (One user, many accounts)
    mule_user = 'User_Mule_King'
    df_dummy.loc[900:950, 'customer_name'] = mule_user
    df_dummy.loc[900:950, 'customer_account_number'] = range(2000, 2051) # 51 unique accounts
    
    print("Running EDA functions...\n")

    # 1. Fan Out
    mules = analyze_fan_out(df_dummy, 'customer_name', 'customer_account_number', threshold=5)
    print("Potential Mules Identified:\n", mules)
    
    # 2. Benford
    analyze_benford(df_dummy, 'amount')
    
    # 3. Round Numbers
    analyze_round_numbers(df_dummy, 'amount')
    
    # 4. Pass Through
    analyze_pass_through(df_dummy, 'customer_account_number', 'amount', 'cashflow_direction')
    
    # 5. Sequence Length
    analyze_sequence_length(df_dummy, 'customer_account_number', 'transaction_date')
    
    # 6. Name Entropy
    analyze_name_entropy(df_dummy, 'customer_name', 'amount')
    
    print("\nEDA demonstration complete.")
