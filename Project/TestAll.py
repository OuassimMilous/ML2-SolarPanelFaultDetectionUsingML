import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import Models

# Define file paths
data_files = ['data.csv']  # List of datasets to combine
results_json_path = os.path.join('logs', 'performance_all_models.json')
mean_plot_path = os.path.join('figures', 'mean_performance_plotboth.png')

# Ensure necessary directories exist
os.makedirs('logs', exist_ok=True)
os.makedirs('tests', exist_ok=True)

# Ensure the results.json file is empty
with open(results_json_path, 'w') as f:
    json.dump({}, f)  # Clear the file or create it if not present

# Function to load and preprocess the dataset
def load_and_preprocess_data(files):
    # Load and concatenate both datasets
    dfs = [pd.read_csv(file) for file in files]
    df = pd.concat(dfs, ignore_index=True)
    
    # Separate numeric columns for imputation
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Handle missing data in numeric columns by replacing with the mean of each column
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Handle categorical 'faulty' column (convert 'No'/'Yes' to 0/1)
    df['faulty'] = df['faulty'].map({'No': 0, 'Yes': 1})
    
    # Define features and target
    X = df.drop(columns=['annotation_file', 'panel_index', 'faulty'])  # Features (exclude irrelevant columns)
    y = df['faulty']  # Target column
    
    # Normalize the data (standard scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# List of models to run (with model names and their corresponding functions)
models = [
    ('Logistic Regression', Models.logistic_regression),
    ('SVM', Models.svm),
    ('KNN', Models.knn),
    ('Random Forest', Models.random_forest),
    ('Gradient Boosting', Models.gradient_boosting),
]

# Initialize results and best parameter counts dictionary to track parameter values
best_param_counts = {
    'Logistic Regression': {'C': {}, 'class_weight': {}, 'penalty': {}, 'solver': {}},
    'SVM': {'C': {}, 'class_weight': {}, 'degree': {}, 'gamma': {}, 'kernel': {}},
    'KNN': {'algorithm': {}, 'leaf_size': {}, 'n_neighbors': {}, 'weights': {}},
    'Random Forest': {'max_depth': {}, 'min_samples_split': {}, 'n_estimators': {}},
    'Gradient Boosting': {'learning_rate': {}, 'loss': {}, 'max_depth': {}, 'min_samples_split': {}, 'n_estimators': {}},
}

# Initialize the dictionary to store model results
all_results = {model[0]: {'F1-Score': [], 'Recall': [], 'Precision': [], 'Accuracy': []} for model in models}

# Load and preprocess data
X_scaled, y = load_and_preprocess_data(data_files)

# Number of folds for K-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True)

# Run the models with K-fold cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(X_scaled), 1):
    print(f"Starting fold {fold} of {k}...")

    # Split data into training and test sets for the current fold
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    for model_name, model_script in models:
        # Run the model and get metrics and best parameters
        accuracy, precision, recall, f1, best_params = model_script(X_train, X_test, y_train, y_test)
        
        # Append results to the results dictionary
        all_results[model_name]['F1-Score'].append(f1)
        all_results[model_name]['Recall'].append(recall)
        all_results[model_name]['Precision'].append(precision)
        all_results[model_name]['Accuracy'].append(accuracy)
        
        # Track unique values and their counts for each hyperparameter
        for param, value in best_params.items():
            if param in best_param_counts[model_name]:
                if value not in best_param_counts[model_name][param]:
                    best_param_counts[model_name][param][value] = 1
                else:
                    best_param_counts[model_name][param][value] += 1
        
        # Save the updated results to the JSON file
        with open(results_json_path, 'w') as f:
            json.dump({'results': all_results, 'best_param_counts': best_param_counts}, f, indent=4)
        
        print(f"{model_name} - Fold {fold}: F1={f1:.2f}, Recall={recall:.2f}, Precision={precision:.2f}, Accuracy={accuracy:.2f}")

print(f"Results saved to {results_json_path}")

# Calculate mean results for each model
mean_results = {model: {metric: np.mean(values) for metric, values in metrics.items()} for model, metrics in all_results.items()}

# Prepare data for plotting
fig, ax = plt.subplots(figsize=(10, 8))

# Bar positions and width
ind = np.arange(len(mean_results))
width = 0.2

# Extract means for each metric
f1_means = [mean_results[model]['F1-Score'] for model in mean_results]
recall_means = [mean_results[model]['Recall'] for model in mean_results]
precision_means = [mean_results[model]['Precision'] for model in mean_results]
accuracy_means = [mean_results[model]['Accuracy'] for model in mean_results]

# Plot bars for metrics (accuracy, precision, recall, F1-score)
bars1 = ax.barh(ind - 1.5 * width, accuracy_means, width, label='Accuracy', color='skyblue')
bars2 = ax.barh(ind - 0.5 * width, precision_means, width, label='Precision', color='orange')
bars3 = ax.barh(ind + 0.5 * width, recall_means, width, label='Recall', color='green')
bars4 = ax.barh(ind + 1.5 * width, f1_means, width, label='F1-Score', color='red')

# Plot best parameter counts
best_param_width = 0.1  # Separate width for best parameters
param_counts = [sum(len(param_dict) for param_dict in best_param_counts[model].values()) for model in best_param_counts]  # Total unique param counts

# Add labels and title
ax.set_xlabel('Scores / Counts')
ax.set_ylabel('Model')
ax.set_title('Model Comparison: Accuracy, Precision, Recall, F1-Score & Best Param Count')
ax.set_yticks(ind)
ax.set_yticklabels(mean_results.keys())

# Add legends
ax.legend(title="Metrics", loc='upper left', bbox_to_anchor=(1, 1))

# Annotate bars with values
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        ax.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}', 
                va='center', ha='right', color='black', fontsize=10)

# Adjust layout
plt.tight_layout()

# Save the mean plot
fig.savefig(mean_plot_path, dpi=300)

# Show plot
plt.show()

print(f"Mean performance plot saved to {mean_plot_path}")
