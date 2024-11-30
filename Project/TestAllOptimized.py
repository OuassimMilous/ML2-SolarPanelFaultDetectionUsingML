import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import ModelsOptimized

# Define file paths for the dataset and results
data_files = ['data.csv']  # List of datasets to combine
results_json_path = os.path.join('logs', 'performance_all_models_optimized.json')
mean_plot_path = os.path.join('figures', 'mean_performance_plotboth_optimized.png')

# Ensure necessary directories for saving results and figures exist
os.makedirs('logs', exist_ok=True)
os.makedirs('tests', exist_ok=True)

# Clear the results.json file or create it if it doesn't exist
with open(results_json_path, 'w') as f:
    json.dump({}, f)  # Create an empty file or clear previous contents

# Function to load and preprocess the dataset
def load_and_preprocess_data(files):
    # Load and concatenate all datasets from the provided list
    dfs = [pd.read_csv(file) for file in files]
    df = pd.concat(dfs, ignore_index=True)
    
    # Identify numeric columns for imputation
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Impute missing values in numeric columns with the column mean
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Convert categorical 'faulty' column to binary values (No=0, Yes=1)
    df['faulty'] = df['faulty'].map({'No': 0, 'Yes': 1})
    
    # Define feature set (X) and target (y)
    X = df.drop(columns=['annotation_file', 'panel_index', 'faulty'])  # Features, excluding irrelevant columns
    y = df['faulty']  # Target column
    
    # Normalize feature data using standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# List of models to run along with their corresponding functions
models = [
    ('Logistic Regression', ModelsOptimized.logistic_regression),
    ('SVM', ModelsOptimized.svm),
    ('KNN', ModelsOptimized.knn),
    ('Random Forest', ModelsOptimized.random_forest),
    ('Gradient Boosting', ModelsOptimized.gradient_boosting),
]

# Initialize a dictionary to track the best hyperparameters for each model
best_param_counts = {
    'Logistic Regression': {'C': {}, 'class_weight': {}, 'penalty': {}, 'solver': {}},
    'SVM': {'C': {}, 'class_weight': {}, 'degree': {}, 'gamma': {}, 'kernel': {}},
    'KNN': {'algorithm': {}, 'leaf_size': {}, 'n_neighbors': {}, 'weights': {}},
    'Random Forest': {'max_depth': {}, 'min_samples_split': {}, 'n_estimators': {}},
    'Gradient Boosting': {'learning_rate': {}, 'loss': {}, 'max_depth': {}, 'min_samples_split': {}, 'n_estimators': {}},
}

# Initialize a dictionary to store performance results for each model
all_results = {model[0]: {'F1-Score': [], 'Recall': [], 'Precision': [], 'Accuracy': []} for model in models}

# Load and preprocess data
X_scaled, y = load_and_preprocess_data(data_files)

# Define number of folds for K-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True)

# Perform K-fold cross-validation for each model
for fold, (train_index, test_index) in enumerate(kf.split(X_scaled), 1):
    print(f"Starting fold {fold} of {k}...")

    # Split data into training and test sets for the current fold
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    for model_name, model_script in models:
        # Run the model and obtain performance metrics and best parameters
        accuracy, precision, recall, f1, best_params = model_script(X_train, X_test, y_train, y_test)
        
        # Append the metrics for the current fold to the results dictionary
        all_results[model_name]['F1-Score'].append(f1)
        all_results[model_name]['Recall'].append(recall)
        all_results[model_name]['Precision'].append(precision)
        all_results[model_name]['Accuracy'].append(accuracy)
        
        # Track the count of each hyperparameter's unique values across all folds
        for param, value in best_params.items():
            if param in best_param_counts[model_name]:
                if value not in best_param_counts[model_name][param]:
                    best_param_counts[model_name][param][value] = 1
                else:
                    best_param_counts[model_name][param][value] += 1
        
        # Save the updated results after each fold to the JSON file
        with open(results_json_path, 'w') as f:
            json.dump({'results': all_results, 'best_param_counts': best_param_counts}, f, indent=4)
        
        # Print model performance for the current fold
        print(f"{model_name} - Fold {fold}: F1={f1:.2f}, Recall={recall:.2f}, Precision={precision:.2f}, Accuracy={accuracy:.2f}")

print(f"Results saved to {results_json_path}")

# Calculate mean performance metrics for each model
mean_results = {model: {metric: np.mean(values) for metric, values in metrics.items()} for model, metrics in all_results.items()}

# Prepare data for plotting the mean performance comparison
fig, ax = plt.subplots(figsize=(10, 8))

# Define positions and width for bars in the plot
ind = np.arange(len(mean_results))
width = 0.2

# Extract mean values for each metric
f1_means = [mean_results[model]['F1-Score'] for model in mean_results]
recall_means = [mean_results[model]['Recall'] for model in mean_results]
precision_means = [mean_results[model]['Precision'] for model in mean_results]
accuracy_means = [mean_results[model]['Accuracy'] for model in mean_results]

# Create horizontal bars for each metric (accuracy, precision, recall, F1-score)
bars1 = ax.barh(ind - 1.5 * width, accuracy_means, width, label='Accuracy', color='skyblue')
bars2 = ax.barh(ind - 0.5 * width, precision_means, width, label='Precision', color='orange')
bars3 = ax.barh(ind + 0.5 * width, recall_means, width, label='Recall', color='green')
bars4 = ax.barh(ind + 1.5 * width, f1_means, width, label='F1-Score', color='red')

# Plot the counts of unique best parameter values
best_param_width = 0.1  # Width for displaying best parameter counts
param_counts = [sum(len(param_dict) for param_dict in best_param_counts[model].values()) for model in best_param_counts]  # Total unique parameter counts

# Add axis labels and title
ax.set_xlabel('Scores / Counts')
ax.set_ylabel('Model')
ax.set_title('Model Comparison: Accuracy, Precision, Recall, F1-Score & Best Param Count')
ax.set_yticks(ind)
ax.set_yticklabels(mean_results.keys())

# Add legends for metrics
ax.legend(title="Metrics", loc='upper left', bbox_to_anchor=(1, 1))

# Annotate each bar with its value for better readability
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        ax.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}', 
                va='center', ha='right', color='black', fontsize=10)

# Adjust layout to avoid overlapping elements
plt.tight_layout()

# Save the generated plot as an image
fig.savefig(mean_plot_path, dpi=300)

# Display the plot
plt.show()

print(f"Mean performance plot saved to {mean_plot_path}")
