import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ### Functions
# Residuals Distribution Histogram (All Models)
def plot_combined_residuals_histogram(df, residual_column='Residuals', model_column='Model', xaxis=5000, ax=None):
    if ax is None:
        ax = plt.gca()

    unique_models = df[model_column].unique()
    for i, model in enumerate(unique_models):
        model_data = df[df[model_column] == model]
        sns.histplot(
            model_data[residual_column], kde=True, bins=200, stat="density",
            label=model, color=f"C{i}", alpha=0.5, ax=ax
        )

    ax.axvline(0, color='red', linestyle='--', linewidth=1, label="Zero Residual")
    ax.set_xlim(-xaxis, xaxis)
    ax.set_title(residual_column.replace('_residuals', '').replace('_', ' ').capitalize())
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Density')

def print_error_table_from_dataframe(df, actual_column, predicted_column, model_column, variab):
    """
    Prints a table summarizing typical errors (MAE, RMSE, MAPE, MSE, R2) for multiple models using a DataFrame.
    
    Parameters:
    - df: DataFrame containing the actual, predicted, and model columns.
    - actual_column: Column name for the actual values.
    - predicted_column: Column name for the predicted values.
    - model_column: Column name for model identifiers.
    
    Returns:
    - None. Prints the error table.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    import pandas as pd

    # Group the DataFrame by the model column
    grouped = df.groupby(model_column)
    
    # Initialize a list to store error metrics for each model
    results = []
    
    # Loop through each group (model)
    for model_name, group in grouped:
        # Extract actual and predicted values
        actual = group[actual_column]
        predicted = group[predicted_column]
        
        # Calculate error metrics
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        # mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        r2 = r2_score(actual, predicted)
        
        # Append the results
        results.append({'Model': model_name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2})
    
    # Create a DataFrame for the results
    error_table = pd.DataFrame(results)

    error_table.to_csv(f"error_table_{variab}.csv", sep=";", decimal=".", encoding="latin1")
    
    # Print the table
    print(error_table)

def get_error_table_from_dataframe(df, actual_column, predicted_column, model_column, variab):
    results = []
    for model in df[model_column].unique():
        df_model = df[df[model_column] == model]
        actual = df_model[actual_column]
        predicted = df_model[predicted_column]
        
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predicted)
        
        results.append({
            'Model': model,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        })
    
    return pd.DataFrame(results)

# Define MAE function
def mae(x):
    return np.mean(np.abs(x))


def round_to_5(x):
    return 5 * round(x / 5)