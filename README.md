# Stroke Prediction using Machine Learning

## Overview
This project aims to predict the likelihood of stroke occurrence based on medical and demographic data. It utilizes machine learning techniques, including SMOTE for handling imbalanced data, stacking classifiers for improved accuracy, and a GUI-based prediction tool using Tkinter.

## Features
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and balancing data.
- **Model Training**: Implements Random Forest, XGBoost, and Logistic Regression-based stacking classifier.
- **Performance Evaluation**: Includes confusion matrix, ROC curve, feature importance visualization, and log loss computation.
- **Graphical User Interface (GUI)**: Allows users to input data and get stroke predictions.

## Dataset
The dataset used is `full_data.csv`, which is preprocessed and balanced to generate `balanced_brainstroke_data.csv`.

## Results
- **Log Loss**: `0.3467`
- **Classification Report**:
  ```
              precision    recall  f1-score   support

           0       0.85      0.82      0.83       283
           1       0.84      0.87      0.85       317

    accuracy                           0.84       600
   macro avg       0.84      0.84      0.84       600
  ```
- **ROC AUC Score**: `0.93`
- **Confusion Matrix**:
  ```
     Predicted 0  Predicted 1
  Actual 0    231          52
  Actual 1     42         275
  ```

## Installation
### Prerequisites
Ensure you have Python installed (preferably Python 3.8+).

### Steps
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/ML-based-Brain-stroke-prediction.git
   cd stroke-prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the prediction model:
   ```sh
   python stroke_prediction.py
   ```

## Usage
- Run the GUI-based prediction tool by executing the script.
- Enter the required medical parameters.
- Click 'Predict' to get stroke probability and classification result.

## Dependencies
See `requirements.txt` for a full list of dependencies.

## License
This project is licensed under the MIT License.

## Author
Shakthi S
