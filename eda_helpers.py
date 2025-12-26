"""
Simple helper functions for exploring Credit Scoring datasets
Interactive table displays using itables for better data exploration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import itables for interactive displays
try:
    from itables import show as ishow
    ITABLES_AVAILABLE = True
except ImportError:
    ITABLES_AVAILABLE = False
    print("⚠️  itables not installed. Install with: pip install itables")


def setup_interactive_display():
    """
    Initialize interactive table display for Jupyter notebooks.
    Call this once at the beginning of your notebook.
    """
    if not ITABLES_AVAILABLE:
        print("❌ itables not available. Interactive display disabled.")
        return
    
    from itables import init_notebook_mode, options
    
    # Initialize itables
    init_notebook_mode(all_interactive=False)
    
    # Configure options
    options.warn_on_undocumented_option = False  # Silence warnings
    options.lengthMenu = [10, 25, 50, 100, 200]
    options.maxBytes = 0
    options.maxColumns = 50
    options.maxRows = 20
    options.paging = True
    options.ordering = True
    options.scrollX = True
    options.scrollY = "400px"
    options.scrollCollapse = True
    options.classes = "display compact"
    options.style = "width:100%"
    
    # Set pandas display options
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.max_colwidth', 100)
    
    print("✓ Interactive scrollable tables enabled")


def _show_table(df, interactive=True):
    """Helper function to display table (interactive or plain)"""
    if interactive and ITABLES_AVAILABLE:
        return ishow(df)
    else:
        print(df.to_string(index=False))


def load_dataset(filename):
    """Load a CSV file from the data directory"""
    data_path = Path('data') / filename
    print(f"Loading {filename}...")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {filename}: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    return df


def dataset_overview(df, name="Dataset", interactive=True):
    """Show dimensions, columns, and types"""
    print(f"{'='*60}")
    print(f"{name.upper()} OVERVIEW")
    print(f"{'='*60}")
    print(f"Dimensions: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nColumn types:")
    print(df.dtypes.value_counts())
    print()
    
    # Create info dataframe
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Non-Null': df.count().values,
        'Null': df.isnull().sum().values,
        'Null %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    
    _show_table(info_df, interactive)
    print()


def missing_values_summary(df, name="Dataset", null_values=None, interactive=True):
    """
    Detailed analysis of missing values
    
    Parameters:
    -----------
    df : DataFrame
        The dataset to analyze
    name : str
        Name of the dataset for display
    null_values : list, optional
        List of additional values to consider as NULL (e.g., ['', 'XNA', 'Unknown'])
    interactive : bool
        Show interactive table (default True)
    
    Returns:
    --------
    tuple : (missing_df, high_missing_cols)
        missing_df: DataFrame with all missing value info
        high_missing_cols: List of columns with >30% missing values
    """
    print(f"{'='*60}")
    print(f"{name.upper()} - MISSING VALUES")
    print(f"{'='*60}")
    
    # Create a copy to work with
    df_check = df.copy()
    
    # Replace custom null values with NaN if provided
    if null_values:
        print(f"Treating as NULL: {null_values}")
        df_check = df_check.replace(null_values, np.nan)
        print()
    
    total_cells = df_check.shape[0] * df_check.shape[1]
    total_missing = df_check.isnull().sum().sum()
    
    print(f"Total cells: {total_cells:,}")
    print(f"Missing cells: {total_missing:,} ({total_missing/total_cells*100:.2f}%)")
    print()
    
    # Columns with missing values
    missing = df_check.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        print("✓ No missing values found!")
        return None, []
    else:
        # Calculate columns by missing percentage thresholds
        missing_pct = (missing.values / len(df_check) * 100)
        cols_above_5 = sum(missing_pct > 5)
        cols_above_15 = sum(missing_pct > 15)
        cols_above_30 = sum(missing_pct > 30)
        
        print(f"Columns with missing values: {len(missing)}/{df_check.shape[1]}")
        print(f"  • >5% missing: {cols_above_5} columns")
        print(f"  • >15% missing: {cols_above_15} columns")
        print(f"  • >30% missing: {cols_above_30} columns")
        print()
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing': missing.values,
            'Missing %': (missing.values / len(df_check) * 100).round(2),
            'Type': df_check[missing.index].dtypes.values
        })
        
        _show_table(missing_df, interactive)
    print()
    
    # Get columns with >30% missing
    high_missing_cols = missing_df[missing_df['Missing %'] > 30]['Column'].tolist()
    
    return missing_df, high_missing_cols


def plot_missing_values(df, name="Dataset", figsize=(12, 6)):
    """Visualize missing values"""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        print("No missing values to plot!")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    missing_pct = (missing / len(df) * 100).head(20)
    axes[0].barh(range(len(missing_pct)), missing_pct.values)
    axes[0].set_yticks(range(len(missing_pct)))
    axes[0].set_yticklabels(missing_pct.index)
    axes[0].set_xlabel('Missing %')
    axes[0].set_title(f'{name} - Top 20 Columns with Missing Values')
    axes[0].invert_yaxis()
    
    # Heatmap (sample if too many rows)
    sample_size = min(1000, len(df))
    if len(df) > sample_size:
        sample_df = df[missing.index].sample(sample_size, random_state=42)
    else:
        sample_df = df[missing.index]
    
    sns.heatmap(sample_df.isnull(), cbar=False, cmap='viridis', 
                yticklabels=False, ax=axes[1])
    axes[1].set_title(f'{name} - Missing Values Pattern\n(Sample of {sample_size} rows)')
    
    plt.tight_layout()
    plt.show()


def numeric_summary(df, name="Dataset", exclude_cols=None, interactive=True):
    """
    Summary statistics for numeric columns
    
    Parameters:
    -----------
    df : DataFrame
        The dataset to analyze
    name : str
        Name of the dataset for display
    exclude_cols : list, optional
        List of column names to exclude from analysis
    interactive : bool
        Show interactive table (default True)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Exclude specified columns
    if exclude_cols:
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        print(f"Excluding {len([c for c in exclude_cols if c in df.columns])} columns from analysis")
    
    print(f"{'='*60}")
    print(f"{name.upper()} - NUMERIC FEATURES")
    print(f"{'='*60}")
    print(f"Number of numeric columns: {len(numeric_cols)}")
    print()
    
    if len(numeric_cols) > 0:
        summary = df[numeric_cols].describe().T
        summary['missing'] = df[numeric_cols].isnull().sum()
        summary['missing_pct'] = (summary['missing'] / len(df) * 100).round(2)
        summary['unique'] = df[numeric_cols].nunique()
        
        # Reset index to make column name a column
        summary = summary.reset_index().rename(columns={'index': 'Column'})
        print()
        
        _show_table(summary, interactive)
    else:
        print("No numeric columns found!")
    print()

def categorical_summary(df, name="Dataset", max_unique=50, exclude_cols=None, interactive=True):
    """
    Summary statistics for categorical columns
    
    Parameters:
    -----------
    df : DataFrame
        The dataset to analyze
    name : str
        Name of the dataset for display
    max_unique : int
        Maximum unique values to show value counts for
    exclude_cols : list, optional
        List of column names to exclude from analysis
    interactive : bool
        Show interactive table (default True)
    """
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Exclude specified columns
    if exclude_cols:
        cat_cols = [col for col in cat_cols if col not in exclude_cols]
        print(f"Excluding {len([c for c in exclude_cols if c in df.columns])} columns from analysis")
    
    print(f"{'='*60}")
    print(f"{name.upper()} - CATEGORICAL FEATURES")
    print(f"{'='*60}")
    print(f"Number of categorical columns: {len(cat_cols)}")
    print()
    
    if len(cat_cols) == 0:
        print("No categorical columns found!")
        return
    
    cat_info = []
    for col in cat_cols:
        cat_info.append({
            'Column': col,
            'Unique': df[col].nunique(),
            'Missing': df[col].isnull().sum(),
            'Missing %': round(df[col].isnull().sum() / len(df) * 100, 2),
            'Top Value': df[col].mode()[0] if len(df[col].mode()) > 0 else None,
            'Top Freq': df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0
        })
    
    cat_df = pd.DataFrame(cat_info)
    print()
    
    _show_table(cat_df, interactive)
    print()
    
    # Show value counts for low cardinality columns
    print(f"\n{'-'*60}")
    print(f"VALUE COUNTS (columns with ≤ {max_unique} unique values)")
    print(f"{'-'*60}\n")
    
    for col in cat_cols:
        if df[col].nunique() <= max_unique:
            print(f"\n{col}:")
            print(df[col].value_counts().to_string())
            print()


def explore_dataset(df, name="Dataset", plot_missing=True):
    """
    Complete exploration of a dataset
    Runs all helper functions in sequence
    """
    dataset_overview(df, name)
    missing_values_summary(df, name)
    
    if plot_missing:
        plot_missing_values(df, name)
    
    numeric_summary(df, name)
    categorical_summary(df, name)


def compare_datasets(*dfs, names=None, interactive=True):
    """Compare multiple datasets side by side"""
    if names is None:
        names = [f"Dataset {i+1}" for i in range(len(dfs))]
    
    print(f"{'='*80}")
    print("DATASETS COMPARISON")
    print(f"{'='*80}\n")
    
    comparison = []
    for df, name in zip(dfs, names):
        comparison.append({
            'Dataset': name,
            'Rows': f"{df.shape[0]:,}",
            'Columns': df.shape[1],
            'Numeric': len(df.select_dtypes(include=[np.number]).columns),
            'Categorical': len(df.select_dtypes(include=['object', 'category']).columns),
            'Missing %': f"{(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%",
            'Memory (MB)': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}"
        })
    
    comp_df = pd.DataFrame(comparison)
    _show_table(comp_df, interactive)
    print()


def find_common_columns(*dfs, names=None, interactive=True):
    """Find common columns across datasets"""
    if names is None:
        names = [f"Dataset {i+1}" for i in range(len(dfs))]
    
    all_columns = [set(df.columns) for df in dfs]
    common = set.intersection(*all_columns)
    
    print(f"{'='*60}")
    print("COMMON COLUMNS ACROSS DATASETS")
    print(f"{'='*60}\n")
    print(f"Common columns: {len(common)}")
    
    if common:
        print(f"\nColumns present in all datasets:")
        for col in sorted(common):
            print(f"  • {col}")
    print()
    
    # Show which columns appear in which datasets
    print(f"\n{'-'*60}")
    print("COLUMN DISTRIBUTION (interactive)")
    print(f"{'-'*60}\n")
    
    all_unique_cols = set.union(*all_columns)
    col_presence = []
    
    for col in sorted(all_unique_cols):
        presence = {name: '✓' if col in df.columns else '–' 
                   for name, df in zip(names, dfs)}
        col_presence.append({'Column': col, **presence})
    
    presence_df = pd.DataFrame(col_presence)
    _show_table(presence_df, interactive)
    print()
    
    return common
