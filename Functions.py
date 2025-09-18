# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import re
from IPython.display import display
from sklearn.preprocessing import LabelEncoder

# Function to clean engine size values based on specific rules
def clean_engine_size(engine_size):
    """
    Cleans engine size values based on specific rules:
    - Returns NaN for missing values or certain non-combustion placeholders.
    - Extracts numeric parts for hybrid cars related to the combustion motor.
    - Returns numeric strings for pure numeric inputs.

    Parameters:
    engine_size (str or float): The engine size value to clean.

    Returns:
    str or float: Cleaned engine size value, or NaN if not applicable.
    """

    # Return NaN if the input is missing
    if pd.isna(engine_size):
        return np.nan
    
    # Convert input to lowercase string and strip spaces for uniform processing
    value = str(engine_size).strip().lower()
    
    # Set of strings considered as non-numeric or electric only, return NaN for these
    nan_cases = {
        '0', '-', 'electric', 'electric motor', 
        'electric (100 kwh)', 'electric (93 kwh)',
        'electric (tri-motor)'
    }
    
    # Return NaN if the value is in the predefined nan_cases set
    if value in nan_cases:
        return np.nan
    
    # If string indicates hybrid or electric or contains '+' (combined),
    # extract the first numeric part found using regex
    if 'hybrid' in value or '+' in value or 'electric' in value:
        match = re.search(r'\d+(\.\d+)?', value)  # Look for a number (int or decimal)
        if match:
            return match.group()  # Return the matched numeric string
        return np.nan  # If no numeric part found, return NaN
    
    # If the value is purely numeric (with optional decimal), return it as is
    if value.replace('.', '').isdigit():
        return value
    
    # For all other cases, return NaN
    return np.nan


# Function to clean non-numeric characters from specified DataFrame columns and to convert common placeholders to NaN
def clean_non_numeric_chars(df, columns, show=False):
    """
    Cleans specified columns in the DataFrame by removing non-numeric characters
    and converting common placeholder values to NaN.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    columns (list or Index): List of column names or indices to clean.
    show (bool): If True, displays rows with unwanted characters before cleaning
                 and the same rows again after cleaning (only for affected columns).

    Returns:
    None: The DataFrame is modified in place.
    """
    for col in columns:
        # Detect non-numeric characters (exclude digits and dot/period)
        unwanted_chars = set()
        for val in df[col].dropna():  # Skip missing values (NaNs)
            # Add all non-digit (0-9) and non-period (.) characters from each value to the set
            unwanted_chars.update(re.findall(r'[^\d\.]', str(val)))

        # Print section headers
        if show:
            print("\n" + "="*40)
            print(f"Processing column: {col}")
            print("="*40)

        if unwanted_chars:
            # Create regex pattern to find any unwanted character
            pattern = '[' + re.escape(''.join(unwanted_chars)) + ']'
            # Filter rows where the column contains any unwanted character
            problem_rows = df[df[col].astype(str).str.contains(pattern, regex=True, na=False)]

            affected_indices = problem_rows.index # Get indices of affected rows

            if show: # If show is True, display the problem rows
                print(f"Before cleaning '{col}', rows with unwanted characters {unwanted_chars}:")
                display(problem_rows) 
        else:
            affected_indices = [] # No unwanted characters found

        # Convert common placeholder values to real NaN
        # This is useful because some cells may contain '-', '' or 'NaN' which should be treated as missing
        # For example Torque (lb-ft) contains '-' or '0' 
        df[col] = df[col].replace(['-', '', '0', 'NaN'], np.nan)

        # Remove each detected unwanted character from the column
        for char in unwanted_chars:
            # Convert only non-null entries to string and replace unwanted character (non-regex)
            mask = df[col].notna()  # Preserve NaNs
            df.loc[mask, col] = df.loc[mask, col].astype(str).str.replace(char, '', regex=False)

        # Print results
        if show:
            if unwanted_chars and len(affected_indices) > 0: # If there were unwanted characters 
                print(f"Cleaned '{col}': removed characters {unwanted_chars}") # Print what was cleaned
                print("After cleaning:")
                display(df.loc[affected_indices]) # Display the same rows after cleaning
            else: # If there were no unwanted characters
                print(f"Nothing to clean in column '{col}'") # print that nothing was cleaned

            


# Function to summarize and display missing values in a DataFrame
def summarize_missing_data(df, show=False):
    """
    Summarizes missing values in the DataFrame and displays:
    - A DataFrame with the count and percentage of missing values per column
    - All rows that contain at least one missing value

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    show (bool): If True, displays rows with missing values.

    Returns:
    None: The function prints the summary and optionally displays rows with missing values.
    """
    
    # Create a DataFrame summarizing missing data in each column:
    # - 'Missing Count' gives the total number of missing (NaN) values per column
    # - 'Missing %' calculates the percentage of missing values per column
    missing_info = pd.DataFrame({
        'Missing Count': df.isnull().sum(),           # Count of NaNs in each column
        'Missing %': df.isnull().mean() * 100         # Percentage of NaNs in each column
    }).sort_values('Missing %', ascending=False)      # Sort columns by descending missing %
    
    # Display the DataFrame showing which columns have missing data and how much
    print("Summary of missing data per column (count and %):")
    display(missing_info)

    if show:
        # Filter and display all rows from df that have at least one missing value in any column
        print("\nRows containing at least one missing value:")
        missing_rows = df[df.isnull().any(axis=1)]       
        display(missing_rows)


# Function to clean specified columns in a DataFrame and convert them to numeric types
def convert_to_numeric(df, columns):
    """
    For the specified columns in df:
    - Convert to string and strip whitespace
    - Convert to numeric (float), coercing errors to NaN
    
    Parameters:
    df (pd.DataFrame): The dataframe to process
    columns (list or Index): List of column names or indices to clean and convert
    
    Returns:
    pd.DataFrame: The dataframe with cleaned and converted columns (in-place modification)
    """
    # Iterate through each specified column
    for col in columns:
        # if the column type is 'object'
        if df[col].dtype == 'object':
            # Convert to string and strip whitespace
            df[col] = df[col].astype(str).str.strip()
    
    # Iterate through each specified column
    for col in columns:
        # Convert to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Return the modified DataFrame
    return df



# Function to classify engine types based on Engine column values
def classify_engine(engine):
    """
    Classifies the type of engine based on the provided engine value.
    - If engine is NaN, '0', or '-', classify as Electric.
    - If 'electric' is present, classify as Electric or Hybrid based on additional keywords.
    - If 'hybrid' is present or '+' symbol, classify as Hybrid.
    - If the string is purely numeric, classify as Combustion.
    - If none of the above conditions match, classify as Unknown.

    Parameters:
    engine (str or float): The engine value to classify.

    Returns:
    str: The classified engine type ('Electric', 'Hybrid', 'Combustion', or 'Unknown').
    """

    # If engine is NaN, '0', or '-', classify as Electric
    if pd.isna(engine) or engine in ['0', '-']:
        return 'Electric'
    
    # Convert to lowercase string for uniform checking and strip whitespace
    engine_str = str(engine).lower().strip()
    
    # Keywords that confirm Electric classification when found alongside 'electric'
    electric_keywords = ['kwh', 'motor', 'tri-motor']
    
    # Check if 'electric' is present in the string
    if 'electric' in engine_str:
        # If any electric-specific keywords are present, classify as Electric
        if any(keyword in engine_str for keyword in electric_keywords):
            return 'Electric'
        
        # Otherwise, check if there is a numeric engine size (e.g., 1.5, 2.0)
        numeric_parts = re.findall(r'[\d\.]+', engine_str)
        if numeric_parts:
            # If numeric size + 'electric' without electric keywords, classify as Hybrid
            return 'Hybrid'
        else:
            # If no numbers but contains 'electric' and no keywords, default to Electric
            return 'Electric'
    
    # Check for Hybrid keywords or patterns
    # If 'hybrid' is in string, or '+' symbol, or '(' with 'hybrid' inside, classify as Hybrid
    if 'hybrid' in engine_str or '+' in engine_str or ('(' in engine_str and 'hybrid' in engine_str):
        return 'Hybrid'
    
    # If the string is purely numeric (can be converted to float), classify as Combustion
    try:
        float(engine_str)
        # Numeric but no electric or hybrid keywords → Combustion
        return 'Combustion'
    except ValueError:
        # If none of the above conditions match, return Unknown
        return 'Unknown'

# Function to remove duplicate rows from a DataFrame
def remove_duplicates(df):
    """
    Checks for duplicate rows in the DataFrame (excluding the first occurrence).
    If duplicates exist, drops them in place and prints summary information.

    Parameters:
    df (pd.DataFrame): The DataFrame to check for duplicates.
    Returns:
    None: The DataFrame is modified in place to remove duplicates.
    """
    
    # Count how many duplicate rows exist (excluding the first occurrence)
    num_duplicates = df.duplicated().sum()
    
    # Print the number of duplicate rows found
    print(f"Number of duplicate rows: {num_duplicates}")
    
    # If any duplicates are found
    if num_duplicates > 0:
        # Drop all duplicates (keep only the first occurrence)
        df.drop_duplicates(inplace=True)
        
        # Print the new shape of the DataFrame after dropping duplicates
        print(f"Duplicates dropped. New shape: {df.shape[0]} rows, {df.shape[1]} columns")
    else:
        # If no duplicates were found, let the user know
        print("No duplicates found. No rows dropped.")


# Function to plot distributions of numeric columns in a DataFrame
def plot_numeric_distributions(df, numeric_cols, width=14, height_per_row=4):
    """
    Plots the distribution of numeric columns in the DataFrame using histograms with KDE overlay.

    Parameters:
    df (pd.DataFrame): The DataFrame containing numeric columns.
    numeric_cols (list): List of numeric column names to plot.
    width (int): Width of the figure.
    height_per_row (int): Height of each row in the subplot grid.

    Returns:
    None: Displays the plots.
    """

    # Define the number of columns for subplot layout
    n_cols = 2
    # Calculate the number of rows needed based on the number of numeric columns
    n_rows = math.ceil(len(numeric_cols) / n_cols)
    
    # Create a grid of subplots with the specified figure size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height_per_row * n_rows))
    # Flatten the 2D axes array to 1D for easier indexing
    axes = axes.flatten()

    # Loop through each numeric column to plot its distribution
    for i, col in enumerate(numeric_cols):
        # Create a histogram with a KDE overlay for the current column
        sns.histplot(df[col], kde=True, bins=30, ax=axes[i])
        # Set title and axis labels for clarity
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    # Remove any unused subplot axes (in case total plots < subplot grid size)
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to prevent overlapping elements
    plt.tight_layout()
    # Display the plot
    plt.show()


# Function to plot boxplots for numeric columns in a DataFrame
def plot_numeric_boxplots(df, numeric_cols, width=14, height_per_row=4):
    """
    Plots boxplots for numeric columns in the DataFrame to visualize their distributions.

    Parameters:
    df (pd.DataFrame): The DataFrame containing numeric columns.
    numeric_cols (list): List of numeric column names to plot.
    width (int): Width of the figure.
    height_per_row (int): Height of each row in the subplot grid.

    Returns:
    None: Displays the boxplots.
    """

    # Define number of columns for the subplot grid
    n_cols = 2
    # Calculate the number of required rows based on the number of numeric columns
    n_rows = math.ceil(len(numeric_cols) / n_cols)

    # Create a grid of subplots with the specified overall figure size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height_per_row * n_rows))
    # Flatten the axes array to make indexing easier when looping
    axes = axes.flatten()

    # Loop through each numeric column and plot a boxplot
    for i, col in enumerate(numeric_cols):
        # Create a boxplot for the current column
        sns.boxplot(x=df[col], ax=axes[i])
        # Set plot title and x-axis label
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].set_xlabel(col)

    # Hide any remaining unused subplots
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to avoid overlapping elements
    plt.tight_layout()
    # Display the plots
    plt.show()



# Function to plot a heatmap of the correlation matrix for numeric columns in a DataFrame
def plot_correlation_heatmap(df, figsize=(10, 8), title="Correlation Heatmap"):
    """
    Plots a heatmap of the correlation matrix for numeric columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    figsize (tuple): Size of the heatmap figure.
    title (str): Title of the plot.
    
    Returns:
    None: Displays the heatmap.
    """
    # Compute the correlation matrix using only numeric columns
    corr_matrix = df.corr(numeric_only=True)

    # Initialize the figure with the specified size
    plt.figure(figsize=figsize)
    
    # Create a heatmap with correlation values annotated inside each cell
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)

    # Set the plot title and apply tight layout to avoid clipping
    plt.title(title)
    plt.tight_layout()
    
    # Display the heatmap
    plt.show()


# Function to apply Label Encoding to specified columns in a DataFrame
def label_encode_columns(df, columns):
    """
    Applies Label Encoding to specified columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing data.
    columns (list): List of column names to encode.

    Returns:
    pd.DataFrame: New DataFrame with encoded columns.
    """
    # Create a copy of the original DataFrame to avoid modifying it directly
    df_encoded = df.copy()

    # Loop through each column specified for encoding
    for col in columns:
        # Proceed only if the column exists in the DataFrame
        if col in df_encoded.columns:
            # Initialize the LabelEncoder
            le = LabelEncoder()
            # Fit the encoder and transform the column values
            df_encoded[col] = le.fit_transform(df_encoded[col])

    # Return the DataFrame with label-encoded columns
    return df_encoded


# Function to generate regression evaluation plots
def regression_evaluation_plots(y_true, y_pred, best_model_name,
                                residuals_vs_pred_xlim=None, residuals_vs_pred_ylim=None,
                                residuals_vs_true_xlim=None, residuals_vs_true_ylim=None,
                                pred_vs_true_xlim=None, pred_vs_true_ylim=None):
    """
    Generates regression evaluation plots:
    1. Residuals vs Predicted Values
    2. Residuals vs True Values
    3. Predicted vs True Values (with perfect prediction line)

    Parameters:
    y_true (array-like): Actual target values (in original scale).
    y_pred (array-like): Predicted target values from the model (in original scale).
    best_model_name (str): Name of the model (used in plot titles).
    *_xlim (tuple or None): x-axis limits for the corresponding plot (min, max).
    *_ylim (tuple or None): y-axis limits for the corresponding plot (min, max).

    Returns:
    None: Displays the plots.
    """
    
    # Compute residuals
    residuals = y_true - y_pred

    # Plot 1: Residuals vs Predicted Values
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values (Original Scale)')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title(f'{best_model_name} Residuals vs Predicted Values (Original Price Scale)')
    if residuals_vs_pred_xlim:
        plt.xlim(residuals_vs_pred_xlim)
    if residuals_vs_pred_ylim:
        plt.ylim(residuals_vs_pred_ylim)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: Residuals vs True Values
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True Values (Original Scale)')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title(f'{best_model_name} Residuals vs True Values (Original Price Scale)')
    if residuals_vs_true_xlim:
        plt.xlim(residuals_vs_true_xlim)
    if residuals_vs_true_ylim:
        plt.ylim(residuals_vs_true_ylim)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 3: Predicted vs True Values
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', lw=2)
    plt.xlabel('True Values (Original Scale)')
    plt.ylabel('Predicted Values (Original Scale)')
    plt.title(f'{best_model_name} Prediction vs True Values (Original Price Scale)')
    if pred_vs_true_xlim:
        plt.xlim(pred_vs_true_xlim)
    if pred_vs_true_ylim:
        plt.ylim(pred_vs_true_ylim)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



        

