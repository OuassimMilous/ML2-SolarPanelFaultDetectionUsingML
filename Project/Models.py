from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Perform hyperparameter tuning using Grid Search with cross-validation
def evaluate_model(model, param_grid, X_train, X_test, y_train, y_test):

    # Perform grid search with 5-fold cross-validation, using F1 score as the evaluation metric
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    # Get the best model from grid search
    best_model = grid_search.best_estimator_
    # Predict on the test set using the best model
    y_pred = best_model.predict(X_test)
    
    # Calculate and store evaluation metrics (all in percentage)
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Confusion Matrix:\n{conf_matrix}\n")
    
    # Return the metrics and best hyperparameters
    return accuracy, precision, recall, f1, grid_search.best_params_

# Logistic Regression Model with Grid Search for Hyperparameter Tuning
def logistic_regression(X_train, X_test, y_train, y_test):
    # Define the parameter grid for logistic regression
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength (inverse of penalty)
        'solver': ['liblinear', 'saga'],  # Optimization algorithms for solving the logistic regression
        'class_weight': [None, 'balanced'],  # Adjust class weights to handle imbalance
        'penalty': ['l1', 'l2', 'elasticnet'],  # Regularization techniques (L1, L2, or ElasticNet)
    }

    model = LogisticRegression()  # Create a logistic regression model
    return evaluate_model(model, param_grid, X_train, X_test, y_train, y_test)

# K-Nearest Neighbors Model with Grid Search for Hyperparameter Tuning
def knn(X_train, X_test, y_train, y_test):
    # Define the parameter grid for KNN (K-Nearest Neighbors)
    param_grid = {
        'n_neighbors': [2, 3, 5, 7, 9],  # Number of neighbors to use in the model
        'weights': ['uniform', 'distance'],  # Weight function used for predictions
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithms for finding nearest neighbors
        'leaf_size': [20, 30, 40],  # Size of leaf nodes for tree-based algorithms
    }
       
    model = KNeighborsClassifier()  # Create a KNN classifier model
    return evaluate_model(model, param_grid, X_train, X_test, y_train, y_test)

# Gradient Boosting Model with Grid Search for Hyperparameter Tuning
def gradient_boosting(X_train, X_test, y_train, y_test):
    # Define the parameter grid for Gradient Boosting model
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of boosting stages (trees)
        'learning_rate': [0.001, 0.01, 0.1, 0.2],  # Step size (learning rate)
        'max_depth': [3, 4, 5, 7],  # Maximum depth of the trees in the boosting process
        'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
        'loss': ['deviance', 'exponential'],  # Loss function used in the boosting process
    }

    model = GradientBoostingClassifier()  # Create a Gradient Boosting classifier model
    return evaluate_model(model, param_grid, X_train, X_test, y_train, y_test)

# Random Forest Model with Grid Search for Hyperparameter Tuning
def random_forest(X_train, X_test, y_train, y_test):
    # Define the parameter grid for Random Forest model
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees in the forest
        'max_depth': [None, 10, 20],  # Maximum depth of the trees
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    }
   
    model = RandomForestClassifier()  # Create a Random Forest classifier model
    return evaluate_model(model, param_grid, X_train, X_test, y_train, y_test)

# Support Vector Machine Model with Grid Search for Hyperparameter Tuning
def svm(X_train, X_test, y_train, y_test):
    # Define the parameter grid for Support Vector Machine (SVM)
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter controlling the trade-off between margin size and classification error
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Type of kernel function to use in SVM
        'gamma': ['scale', 'auto'],  # Kernel coefficient for non-linear kernels
        'class_weight': [None, 'balanced'],  # Adjust class weights to handle imbalance
        'degree': [2, 3, 4],  # Degree of polynomial kernel (relevant for 'poly' kernel)
    }

    model = SVC()  # Create an SVM classifier model
    return evaluate_model(model, param_grid, X_train, X_test, y_train, y_test)
