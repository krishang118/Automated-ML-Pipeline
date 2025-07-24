# Automated ML Pipeline
  
A comprehensive and robust end-to-end Automated Machine Learning pipeline designed to seamlessly streamline the process of data loading and exploration, preprocessing, feature engineering, model training, evaluation, interpretation, and prediction for both classification and regression tasks, with minimal manual intervention.

## Project Flow and Features

- Data Loading: Supports multiple file formats (csv, excel, json, parquet) with automatic encoding and separator detection.
- Exploratory Data Analysis (EDA): Comprehensive data analysis including missing values, data types, statistical summaries, correlations, and target variable analysis with visualizations.
- Data Preprocessing: Handles missing values (KNN or mean/median imputation), outliers (flagging or capping), categorical encoding (label or one-hot), and feature scaling.
- Feature Engineering: Creates interaction, polynomial, and PCA-derived features, with automated feature selection based on correlation, mutual information, or statistical tests.
- Model Training: Supports multiple algorithms for classification and regression, including Random Forest, XGBoost, LightGBM, Logistic/Linear Regression, SVM, and more, with optional hyperparameter tuning via Optuna.
- Results Analysis: Generates detailed performance reports, model comparisons, feature importance, and visualizations like confusion matrix, ROC curve, residual plots and more.
- Model Interpretation: Provides insights into model performance, feature importance, and prediction quality.
- Predictions and Deployment: Allows predictions on new data using the trained pipeline, with support for exporting the model.

## How to Run

### Requirements

- Python 3.8+
- Install the required dependencies:
```bash
pip install pandas numpy matplotlib scikit-learn seaborn scipy optuna xgboost lightgbm imbalanced-learn openpyxl pyarrow
```

### Usage

1. Clone this repository on your local machine.
2. Prepare the dataset file (CSV, Excel, JSON, or Parquet) with a clear target column.
3. Open and edit the main block in the `Automated ML Pipeline.ipynb` Jupyter Notebook to specify your data path and target column:
```python
data_path = "regression_data.csv"  # Use your dataset file (the repository does contain test datasets too)
target_column = "target"           # Change as needed
```
4. Run and execute the `Automated ML Pipeline.ipynb` Jupyter Notebook (making sure the dataset file lies in the same directory).
5. Results:
   - Detailed reports and plots are displayed.
   - Predictions are saved to `predictions.csv`.
   - The trained pipeline is saved as `automl_pipeline.pkl` and may be utilized in future workflows.

- The repository includes testing datasets (`regression_data.csv`, `classification_data.csv`) and scripts for creating them (`test_regression.py`, `test_classification.py`).
- The pipeline is highly configurable and customizable as well, and the parameters can be changed and adjusted by the user if required; especially by modifying and toggling the thresholds and options in the `Config` class and the `run_pipeline` arguments.

### Supported Models

```
For Regression: Linear Regression, Lasso Regression, Ridge Regression, Elastic Net, K-Neighbors, SVM, Random Forest, Decision Tree, Gradient Boosting, XGBoost, LightGBM
For Classification: Logistic Regression, Naive Bayes, K-Neighbors, SVM, Random Forest, Decision Tree, Gradient Boosting, XGBoost, LightGBM
```

## Acknowledgements

Special thanks to the Indian Cybercrime Coordination Centre (I4C), Ministry of Home Affairs (MHA), for supporting this work.

## Contributing

Contributions are welcome!

## License

Distributed under the MIT License. 
