
# Solar Panel Fault Detection via Drone-Captured Images

## Overview

This project automates the fault detection process in solar photovoltaic (PV) systems using drone-captured thermal images. By leveraging machine learning models, the goal is to identify defective modules based on temperature anomalies captured by infrared cameras.

## Problem Statement

Solar PV systems are prone to module failures due to operational stresses and installation errors. Traditional inspection methods are time-consuming and labor-intensive. This project addresses these challenges by developing a machine learning pipeline to analyze drone-captured thermal images and identify defective modules efficiently.

## Dataset

The dataset, sourced from Kaggle, consists of:
- **Thermal (IR) images**: Captured by drones.
- **Annotations**: Marking the corners and centers of each PV module and labeling them as "defective" or "non-defective."

### Challenges
- **Imbalanced Data**: Few examples of faulty modules compared to non-faulty ones.
- **Limited Thermal Data**: Addressed using feature extraction techniques.

## Features Extracted
- **Temperature Statistics**: Mean, standard deviation, maximum, and range.
- **Distribution Metrics**: Skewness and kurtosis.
- **Fault Detection**: Labels from annotations indicating module condition.

<img src="https://github.com/OuassimMilous/ML2---Solar-Panel-Fault-Detection-using-ML/raw/main/Project/figures/Features%20Plot.png" width="500"/>

## Tested Machine Learning Models
The following models were tested:
1. **Support Vector Machine (SVM)**
2. **Random Forest Classifier**
3. **Gradient Boosting Classifier**
4. **K-Nearest Neighbors (KNN)**
5. **Logistic Regression**


### Evaluation Metric
Due to the imbalanced dataset, the **F1-Score** was chosen as the primary evaluation metric, ensuring a balance between precision and recall.

## Hyperparameter Tuning
Each model was optimized using GridSearchCV with 5-fold cross-validation. The best parameters for each model were:

<img src="https://github.com/OuassimMilous/ML2---Solar-Panel-Fault-Detection-using-ML/blob/main/Project/figures/hyperparameter_performance.png"/>

### Logistic Regression
- **C**: 10
- **Solver**: liblinear
- **Class Weight**: None
- **Penalty**: l1

### SVM (Support Vector Machine)
- **C**: 0.01
- **Kernel**: Linear
- **Gamma**: Scale
- **Class Weight**: Balanced
- **Degree**: 2

### K-Nearest Neighbors (KNN)
- **Algorithm**: auto
- **Leaf Size**: 20
- **n_neighbors**: 3
- **Weights**: uniform

### Random Forest
- **Max Depth**: None
- **Min Samples Split**: 2
- **n_estimators**: 200

### Gradient Boosting
- **Learning Rate**: 0.1
- **Loss**: Exponential
- **Max Depth**: 5
- **Min Samples Split**: 2
- **n_estimators**: 50

## Performance Overview
Each model was optimized using GridSearchCV with 5-fold cross-validation.
<img src="https://github.com/OuassimMilous/ML2---Solar-Panel-Fault-Detection-using-ML/blob/main/Project/figures/mean_performance_plotboth_optimized.png"/>

### Logistic Regression
- **F1-Score**: [66.67, 86.96, 77.78, 78.26, 88.00]
- **Recall**: [53.33, 83.33, 70.00, 64.29, 84.62]
- **Precision**: [88.89, 90.91, 87.50, 100.00, 91.67]
- **Accuracy**: [97.07, 98.90, 98.53, 98.16, 98.90]

### SVM (Support Vector Machine)
- **F1-Score**: [100.00, 81.48, 84.21, 76.92, 88.00]
- **Recall**: [100.00, 91.67, 80.00, 71.43, 84.62]
- **Precision**: [100.00, 73.33, 88.89, 83.33, 91.67]
- **Accuracy**: [100.00, 98.17, 98.90, 97.79, 98.90]

### K-Nearest Neighbors (KNN)
- **F1-Score**: [80.00, 76.19, 70.59, 80.00, 91.67]
- **Recall**: [66.67, 66.67, 60.00, 71.43, 84.62]
- **Precision**: [100.00, 88.89, 85.71, 90.91, 100.00]
- **Accuracy**: [98.17, 98.17, 98.16, 98.16, 99.26]

### Random Forest
- **F1-Score**: [63.64, 81.82, 84.21, 78.26, 78.26]
- **Recall**: [46.67, 75.00, 80.00, 64.29, 69.23]
- **Precision**: [100.00, 90.00, 88.89, 100.00, 90.00]
- **Accuracy**: [97.07, 98.53, 98.90, 98.16, 98.16]

### Gradient Boosting
- **F1-Score**: [75.00, 81.82, 66.67, 80.00, 81.82]
- **Recall**: [60.00, 75.00, 60.00, 71.43, 69.23]
- **Precision**: [100.00, 90.00, 75.00, 90.91, 100.00]
- **Accuracy**: [97.80, 98.53, 97.79, 98.16, 98.53]

we clearly notice **SVM** outperforming the other algorithms.

## Results
The SVM (Support Vector Machine) model achieved the best performance among all tested models, using the following optimal hyperparameters:

### SVM (Support Vector Machine)
- **C**: 0.01
- **Kernel**: Linear
- **Gamma**: Scale
- **Class Weight**: Balanced
- **Degree**: 2

Upon testing, the model yielded the following average performance metrics:
![SVM Performance Plot](https://github.com/OuassimMilous/ML2---Solar-Panel-Fault-Detection-using-ML/blob/main/Project/figures/mean_performance_plot_final.png)

- **Accuracy**: 98.60%
- **F1-Score**: 86.11%
- **Precision**: 86.53%
- **Recall**: 87.56%



## Acknowledgments
- Kaggle dataset "Photovoltaic system thermography" by Marcos Gabriel. [link](https://www.kaggle.com/datasets/marcosgabriel/photovoltaic-system-thermography/data)
- Infrared thermal imaging for fault detection in solar panels dataset by [Technodivesh](https://github.com/technodivesh/imagesearch/tree/master/images)


---

This project was made by Ouassim Milous as part of the **Learning for Robotics II** course by professor Luca Oneto at the University of Genoa.
