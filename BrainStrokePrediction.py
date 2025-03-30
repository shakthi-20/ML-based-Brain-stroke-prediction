#hv to edit roc auc curve and display confusion matrix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score,roc_curve
from sklearn.metrics import ConfusionMatrixDisplay


# Load Dataset
data = pd.read_csv(r"D:\ml(uni)\balanced_brainstroke_data.csv")
data.replace('N/A', np.nan, inplace=True)
data['bmi'] = pd.to_numeric(data['bmi'], errors='coerce')
data.loc[:, 'bmi'] = data['bmi'].fillna(data['bmi'].mean())

data = pd.get_dummies(data, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)

X = data.drop(['stroke'], axis=1)
y = data['stroke']

# Handle Class Imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Standardization

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()  # Normalize between 0 and 1
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save feature names
feature_names = X.columns.tolist()
with open('feature_names.pkl', 'wb') as feature_file:
    pickle.dump(feature_names, feature_file)

# Train Base Models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
pos_weight = len(y_resampled[y_resampled == 0]) / len(y_resampled[y_resampled == 1])
xgb_model = XGBClassifier(eval_metric='logloss', scale_pos_weight=pos_weight, learning_rate=0.05, max_depth=5)


rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Define Base Models
base_models = [
    ('rf', rf_model),
    ('xgb', xgb_model)
]

# Meta-Model for Stacking
meta_model = LogisticRegression(max_iter=1000, class_weight="balanced")
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, passthrough=True)

# Train Stacked Model
print("\n Training Stacked Model...")
stacking_model.fit(X_train, y_train)

# Load feature names
with open('feature_names.pkl', 'rb') as feature_file:
    feature_names = pickle.load(feature_file)

# Evaluate Model
y_pred = stacking_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
auc = roc_auc_score(y_test, stacking_model.predict_proba(X_test)[:, 1])
##


# Random Forest Feature Importance
rf_importances = rf_model.feature_importances_
xgb_importances = xgb_model.feature_importances_

# Plot Feature Importance
plt.figure(figsize=(10, 5))
plt.barh(feature_names, rf_importances, color="skyblue")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance (Random Forest)")
plt.show()

plt.figure(figsize=(10, 5))
plt.barh(feature_names, xgb_importances, color="lightcoral")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance (XGBoost)")
plt.show()

# Evaluate Model
y_pred = stacking_model.predict(X_test)
y_prob = stacking_model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
from sklearn.metrics import log_loss
# Ensure y_pred_proba contains probability estimates (not class labels)
log_loss_value = log_loss(y_test, y_prob)
print(f"Log Loss: {log_loss_value:.4f}")


# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
from sklearn.metrics import classification_report

# Print classification report
y_pred_labels = (y_prob >= 0.5).astype(int)  # Convert probabilities to binary predictions
print(classification_report(y_test, y_pred_labels))

import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
corr_matrix = X.corr()

####


##



# Tkinter GUI
root = tk.Tk()
root.title("Stroke Prediction App")
root.geometry("500x700")
root.configure(bg='#f0f0f0')

tk.Label(root, text="Stroke Prediction Form", font=("Arial", 16, "bold"), bg='#f0f0f0').pack(pady=10)

fields = ["Age", "Glucose Level", "BMI"]
entries = {}
for field in fields:
    tk.Label(root, text=field + ":", bg='#f0f0f0').pack()
    entry = tk.Entry(root)
    entry.pack()
    entries[field] = entry

tk.Label(root, text="Hypertension:", bg='#f0f0f0').pack()
hypertension_var = tk.StringVar(value="0")
ttk.Combobox(root, textvariable=hypertension_var, values=["0: No", "1: Yes"]).pack()

tk.Label(root, text="Heart Disease:", bg='#f0f0f0').pack()
heart_disease_var = tk.StringVar(value="0")
ttk.Combobox(root, textvariable=heart_disease_var, values=["0: No", "1: Yes"]).pack()

tk.Label(root, text="Smoking Status:", bg='#f0f0f0').pack()
smoking_var = tk.StringVar(value="0")
ttk.Combobox(root, textvariable=smoking_var, values=["0: Never Smoked", "1: Formerly Smoked", "2: Smokes"]).pack()

tk.Label(root, text="Gender:", bg='#f0f0f0').pack()
gender_var = tk.StringVar(value="0")
ttk.Combobox(root, textvariable=gender_var, values=["0: Female", "1: Male"]).pack()

tk.Label(root, text="Residence Type:", bg='#f0f0f0').pack()
residence_var = tk.StringVar(value="0")
ttk.Combobox(root, textvariable=residence_var, values=["0: Rural", "1: Urban"]).pack()

tk.Label(root, text="Work Type:", bg='#f0f0f0').pack()
work_var = tk.StringVar(value="0")
ttk.Combobox(root, textvariable=work_var, values=["0: Private", "1: Self-Employed", "2: Children"]).pack()

def predict_stroke():
    try:
        user_data = {
            "Age": float(entries['Age'].get()),
            "Glucose Level": float(entries['Glucose Level'].get()),
            "BMI": float(entries['BMI'].get()),
            "Hypertension": int(hypertension_var.get().split(":")[0]),
            "Heart Disease": int(heart_disease_var.get().split(":")[0]),
            "Smoking Status": int(smoking_var.get().split(":")[0]),
            "Gender": int(gender_var.get().split(":")[0]),
            "Residence Type": int(residence_var.get().split(":")[0]),
            "Work Type": int(work_var.get().split(":")[0])
        }
        
        for col in feature_names:
            if col not in user_data:
                user_data[col] = 0
        
        features = np.array([user_data[col] for col in feature_names]).reshape(1, -1)
        features = scaler.transform(features)
        prediction = stacking_model.predict(features)[0]
        probability = stacking_model.predict_proba(features)[0][1]
        
        messagebox.showinfo("Prediction Result", f"Stroke Prediction: {int(prediction)}\nProbability: {probability:.4f}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

tk.Button(root, text="Predict", command=predict_stroke, font=("Arial", 12, "bold"), bg='#4CAF50', fg='white', padx=20, pady=5).pack(pady=20)

root.mainloop()
