import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter # Ensure this is installed: pip install xlsxwriter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os
try:
    df = pd.read_excel("/Users/venkatchandan/Desktop/sensor_calibration_project/Main_gate/Main_gate_exp2.xlsx")
except FileNotFoundError:
    print(" not found. Please ensure the file is in the correct directory.")
    exit()


time_col = 'Time'
reference_col = 'Reference monitor' # This is the target variable
# For now, we are focusing only on 'Honeywell_01' as requested.
# When you ask to expand, this list will be dynamically generated again.
sensor_cols_to_process = ['Honeywell_01','Honeywell_03', 'Honeywell_04','Plantower_01', 'Plantower_03','Plantower_04','SPS30_01','SPS30_03', 'SPS30_04']
y_full = df[reference_col]

# ====== METRIC FUNCTION ======
def compute_metrics(y_true, y_pred):
    """
    Computes Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2).

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

# ====== FOR EACH SPECIFIED SENSOR COLUMN ======
# The loop will currently run only for 'Honeywell_01'
for sensor in sensor_cols_to_process:
    print(f"🔍 Processing sensor: {sensor}")

    # Define the output Excel file name for the current sensor.
    excel_file_name = f"{sensor}_prediction.xlsx"
    # Initialize an Excel writer for each sensor, creating a separate file for each.
    writer = pd.ExcelWriter(excel_file_name, engine="xlsxwriter")

    # Extract the current sensor's data as the feature (X).
    X = df[[sensor]]
    # Split the data into training and testing sets (80% train, 20% test).
    X_train, X_test, y_train, y_test = train_test_split(X, y_full, test_size=0.2, random_state=42)

    # List to store temporary plot file paths for cleanup
    plot_files_to_clean = []

    # --- Helper function to process each model ---
    def process_model(model_name, model_estimator, param_grid, X_train, y_train, X_test, y_test, X_full, y_full, writer):
        """
        Trains a model with GridSearchCV, makes predictions, computes metrics,
        and writes results including hyperparameters and a plot to Excel.
        """
        print(f"  Training {model_name} for {sensor}...")

        # Perform GridSearchCV for hyperparameter tuning
        # cv=5 for 5-fold cross-validation, scoring='r2' to optimize R-squared.
        # n_jobs=-1 uses all available CPU cores for faster computation.
        grid_search = GridSearchCV(model_estimator, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Predict on the full dataset (X_full) and the test dataset (X_test)
        pred_full = best_model.predict(X_full)
        pred_test = best_model.predict(X_test)

        # Compute metrics for both full and test predictions
        metrics_full = compute_metrics(y_full, pred_full)
        metrics_test = compute_metrics(y_test, pred_test)

        # --- Write Predictions and Metrics to Excel ---
        # Full data predictions
        pd.DataFrame({'Actual': y_full.reset_index(drop=True), 'Predicted': pred_full}).to_excel(
            writer, sheet_name=f'{model_name}_Predictions_Full', index=False
        )
        pd.DataFrame([metrics_full]).to_excel(
            writer, sheet_name=f'{model_name}_Metrics_Full', index=False
        )

        # Test data predictions
        pd.DataFrame({'Actual': y_test.reset_index(drop=True), 'Predicted': pred_test}).to_excel(
            writer, sheet_name=f'{model_name}_Predictions_Test', index=False
        )
        pd.DataFrame([metrics_test]).to_excel(
            writer, sheet_name=f'{model_name}_Metrics_Test', index=False
        )

        # --- Write Hyperparameter Details and Embed Plot ---
        # Create a DataFrame from GridSearchCV results for all parameters tried
        cv_results_df = pd.DataFrame(grid_search.cv_results_)
        # Create a DataFrame for the best parameters found
        best_params_df = pd.DataFrame([grid_search.best_params_])

        # Get the workbook and worksheet objects for plot embedding
        workbook = writer.book
        hyperparams_sheet_name = f'{model_name}_Hyperparameters'
        hyperparams_sheet = workbook.add_worksheet(hyperparams_sheet_name)

        # Write best parameters to the sheet
        hyperparams_sheet.write('A1', 'Best Hyperparameters:')
        best_params_df.to_excel(writer, sheet_name=hyperparams_sheet_name, startrow=1, startcol=0, index=False)

        # Write all CV results to the sheet, starting after best params
        hyperparams_sheet.write(len(best_params_df) + 4, 0, 'All Cross-Validation Results:')
        cv_results_df.to_excel(writer, sheet_name=hyperparams_sheet_name, startrow=len(best_params_df) + 5, startcol=0, index=False)

        # --- Generate and Embed Plot ---
        plt.figure(figsize=(10, 6)) # Set figure size for better visibility
        plt.scatter(y_full, pred_full, alpha=0.6) # Scatter plot of actual vs. predicted values
        plt.plot([y_full.min(), y_full.max()], [y_full.min(), y_full.max()], 'r--', lw=2) # Add a perfect prediction line
        plt.title(f'{model_name} - Actual vs. Predicted for {sensor} (Full Data)')
        plt.xlabel('Actual Reference Monitor Value')
        plt.ylabel(f'Predicted Reference Monitor Value by {model_name}')
        plt.grid(True)
        plt.tight_layout() # Adjust layout to prevent labels from overlapping

        # Save the plot to a temporary PNG file
        plot_filename = f'temp_plot_{sensor}_{model_name}.png'
        plt.savefig(plot_filename)
        plot_files_to_clean.append(plot_filename) # Add to list for cleanup
        plt.close() # Close the plot to free up memory

        # Insert the image into the Excel sheet
        # Adjust cell 'G2' as needed to position the image appropriately
        hyperparams_sheet.insert_image('G2', plot_filename)

        return metrics_test['R2'] # Return test R2 for logging

    # ==== RANDOM FOREST REGRESSOR ====
    rf_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf_estimator = RandomForestRegressor(random_state=42)
    rf_r2 = process_model('RF', rf_estimator, rf_grid, X_train, y_train, X_test, y_test, X, y_full, writer)

    # ==== XGBOOST REGRESSOR ====
    xgb_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.1],
        'subsample': [1.0],
        'colsample_bytree': [1.0]
    }
    xgb_estimator = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    xgb_r2 = process_model('XGB', xgb_estimator, xgb_grid, X_train, y_train, X_test, y_test, X, y_full, writer)

    # ==== SVR ====
    # SVR requires scaling, so we use a pipeline.
    svr_pipeline = make_pipeline(StandardScaler(), SVR())
    svr_grid = {
        'svr__C': [1, 10, 100],
        'svr__gamma': [0.01, 0.1],
        'svr__epsilon': [0.01, 0.1],
        'svr__kernel': ['rbf']
    }
    svr_r2 = process_model('SVR', svr_pipeline, svr_grid, X_train, y_train, X_test, y_test, X, y_full, writer)

    # LOGGING for the current sensor
    print(f"✅ {sensor} :: RF R²: {rf_r2:.4f}, XGB R²: {xgb_r2:.4f}, SVR R²: {svr_r2:.4f}")

    # ====== SAVE EXCEL FILE ======
    writer.close()
    print(f"✅ Results for {sensor} written to {excel_file_name}.")

    # ====== CLEAN UP TEMPORARY PLOT FILES ======
    for plot_file in plot_files_to_clean:
        if os.path.exists(plot_file):
            os.remove(plot_file)
            # print(f"Cleaned up {plot_file}")

print("✅ All specified sensor models processed.")
