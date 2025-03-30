import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("full_data.csv")

# Encode categorical columns using LabelEncoder
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store the encoders if needed for inverse transformation

# Features (X) and Target (y)
X = df.drop(columns=['stroke'])  # Drop the stroke column
y = df['stroke']  # Target variable

# **Step 1: Use SMOTE to increase stroke cases (1) from 200 to 500**
smote = SMOTE(sampling_strategy=500/len(df[df['stroke'] == 0]), random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Convert back to DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['stroke'] = y_resampled

# **Step 2: Downsample the majority class (reduce 0 cases)**
majority_class = df[df['stroke'] == 0]  # Select all "No Stroke" cases
majority_downsampled = majority_class.sample(n=1000, random_state=42)  # Reduce to 1000 cases

# **Step 3: Combine balanced data**
final_df = pd.concat([df_resampled[df_resampled['stroke'] == 1], majority_downsampled])

# Save the balanced dataset
final_df.to_csv("balanced_brainstroke_data.csv", index=False)

print("Balanced dataset created successfully!")
