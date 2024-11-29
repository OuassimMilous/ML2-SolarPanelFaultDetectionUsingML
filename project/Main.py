import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import Models

# Define file paths for saving logs and plot
plot_path = os.path.join('figures', 'Performance plot.png')
file_path = 'data.csv'
log_file_path = os.path.join('logs', 'Performance Logs.txt')

# Function to load and preprocess the dataset
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)  # Load the dataset into a DataFrame
    
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
    
    # Split the dataset into training and testing sets
    return train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# List of models to run (with model names and their corresponding functions)
models = [
    ('Logistic Regression', Models.logistic_regression),
    ('SVM', Models.svm),
    ('KNN', Models.knn),
    ('Random Forest', Models.random_forest),
    ('Gradient Boosting', Models.gradient_boosting),
]

# Dictionary to store the results of all models
results = {model_name: {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': [], 'Best Parameters': []} for model_name, _ in models}

# Ensure the 'logs' directory exists for storing performance logs
os.makedirs('logs', exist_ok=True)

# Open the log file for writing (dynamic updates during testing)
with open(log_file_path, 'w') as log_file: 
    log_file.write(f"Testing with dataset: {file_path}\n")  # Write the dataset name to the log
    print(f"Testing with dataset: {file_path}")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

    # Loop through the models and store the results for each one
    for model_name, model_script in models:
        log_file.write(f"Testing {model_name}...\n")  # Log the model being tested
        print(f"Testing {model_name}...")
        
        # Run the model and get detailed metrics (accuracy, precision, recall, etc.)
        accuracy, precision, recall, f1, best_params = model_script(X_train, X_test, y_train, y_test)
        
        # Store the results for each model in the results dictionary
        results[model_name]['Accuracy'].append(accuracy)
        results[model_name]['Precision'].append(precision)
        results[model_name]['Recall'].append(recall)
        results[model_name]['F1-Score'].append(f1)
        results[model_name]['Best Parameters'].append(best_params)
        
        # Log the metrics immediately after testing the model
        log_file.write(f"{model_name} Accuracy: {accuracy:.2f}%\n")
        log_file.write(f"{model_name} Precision: {precision:.2f}%\n")
        log_file.write(f"{model_name} Recall: {recall:.2f}%\n")
        log_file.write(f"{model_name} F1-Score: {f1:.2f}%\n")
        log_file.write(f"{model_name} Best Parameters: {best_params}\n\n")
        log_file.flush()  # Ensure the log file is updated immediately
        
        # Printing the metrics        
        print(f"{model_name} Accuracy: {accuracy:.2f}%")
        print(f"{model_name} Precision: {precision:.2f}%")
        print(f"{model_name} Recall: {recall:.2f}%")
        print(f"{model_name} F1-Score: {f1:.2f}%")
        print(f"{model_name} Best Parameters: {best_params}\n")

# Print log file location for user reference
print(f"Logs saved to {log_file_path}")

# Ensure the 'figures' directory exists for saving the performance plot
os.makedirs('figures', exist_ok=True)

# Plotting results for comparison (bar plot for accuracy, precision, recall, and F1-Score)
fig, ax = plt.subplots(figsize=(10, 8))

# Prepare data for bar plot
model_names = list(results.keys())
accuracy_values = [result['Accuracy'][0] for result in results.values()]
precision_values = [result['Precision'][0] for result in results.values()]
recall_values = [result['Recall'][0] for result in results.values()]
f1_values = [result['F1-Score'][0] for result in results.values()]

# Bar positions and bar width for better visualization
ind = np.arange(len(model_names))  # x locations for the models
width = 0.2  # Width of each bar in the plot

# Plot bars for each metric (accuracy, precision, recall, F1-score)
bars1 = ax.barh(ind - 1.5 * width, accuracy_values, width, label='Accuracy', color='skyblue')
bars2 = ax.barh(ind - 0.5 * width, precision_values, width, label='Precision', color='orange')
bars3 = ax.barh(ind + 0.5 * width, recall_values, width, label='Recall', color='green')
bars4 = ax.barh(ind + 1.5 * width, f1_values, width, label='F1-Score', color='red')

# Set axis labels, title, and ticks
ax.set_xlabel('Scores')
ax.set_ylabel('Model')
ax.set_title('Model Comparison: Accuracy, Precision, Recall, F1-Score')
ax.set_yticks(ind)
ax.set_yticklabels(model_names)

# Add legends to differentiate the metrics
ax.legend(title="Metrics", loc='upper left', bbox_to_anchor=(1, 1))

# Annotate each bar with its corresponding metric value
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        ax.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}', 
                va='center', ha='right', color='black', fontsize=10)

# Adjust layout to prevent overlapping text and labels
plt.tight_layout()

# Save the plot as a PNG file
fig.savefig(plot_path, dpi=300)

# Optionally display the plot
plt.show()

# Print save location of the plot
print(f"Plot saved to {plot_path}")
