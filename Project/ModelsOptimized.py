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
        'C': [10],  # Regularization strength (inverse of penalty)
        'solver': ['liblinear'],  # Optimization algorithms
        'class_weight': [None],  # Handle class imbalance
        'penalty': ['l1'],  # Regularization type (L1, L2, or ElasticNet)
    }

    model = LogisticRegression()  # Create a logistic regression model
    return evaluate_model(model, param_grid, X_train, X_test, y_train, y_test)

# K-Nearest Neighbors Model with Grid Search for Hyperparameter Tuning
def knn(X_train, X_test, y_train, y_test):
    # Define the parameter grid for KNN
    param_grid = {
        'n_neighbors': [3],  # Number of neighbors to use
        'weights': ['uniform'],  # Weight function for predictions
        'algorithm': ['auto'],  # Nearest neighbor search algorithm
        'leaf_size': [20],  # Leaf size for tree-based algorithms
    }
    
    model = KNeighborsClassifier()  # Create a KNN classifier model
    return evaluate_model(model, param_grid, X_train, X_test, y_train, y_test)

# Gradient Boosting Model with Grid Search for Hyperparameter Tuning
def gradient_boosting(X_train, X_test, y_train, y_test):
    # Define the parameter grid for Gradient Boosting
    param_grid = {
        'n_estimators': [50],  # Number of boosting stages
        'learning_rate': [0.1],  # Step size
        'max_depth': [5],  # Maximum depth of individual trees
        'min_samples_split': [2],  # Minimum samples required to split a node
        'loss': ['exponential'],  # Loss function for optimization
    }

    model = GradientBoostingClassifier()  # Create a Gradient Boosting model
    return evaluate_model(model, param_grid, X_train, X_test, y_train, y_test)

# Random Forest Model with Grid Search for Hyperparameter Tuning
def random_forest(X_train, X_test, y_train, y_test):
    # Define the parameter grid for Random Forest
    param_grid = {
        'n_estimators': [200],  # Number of trees in the forest
        'max_depth': [None],  # Maximum depth of trees
        'min_samples_split': [2],  # Minimum samples required to split a node
    }
    
    model = RandomForestClassifier()  # Create a Random Forest classifier model
    return evaluate_model(model, param_grid, X_train, X_test, y_train, y_test)

# Support Vector Machine Model with Grid Search for Hyperparameter Tuning
def svm(X_train, X_test, y_train, y_test):
    # Define the parameter grid for SVM
    param_grid = {
        'C': [0.01],  # Regularization parameter
        'kernel': ['linear'],  # Kernel types
        'gamma': ['scale'],  # Kernel coefficient for non-linear kernels
        'class_weight': ["balanced"],  # Handle class imbalance
        'degree': [2],  # Degree for polynomial kernel
    }

    model = SVC()  # Create an SVM model
    return evaluate_model(model, param_grid, X_train, X_test, y_train, y_test)
