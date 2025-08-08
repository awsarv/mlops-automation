import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

# Pull the latest data from DVC
os.system("dvc pull data/california_housing.csv")
df = pd.read_csv('data/california_housing.csv')

# Impute missing values
imputer = SimpleImputer(strategy='mean')
features = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Encode categorical variables if present
cat_cols = features.select_dtypes(include=['object', 'category']).columns
if len(cat_cols) > 0:
    features = pd.get_dummies(features, columns=cat_cols, drop_first=True)

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_df = pd.DataFrame(features_scaled, columns=features.columns)

try:
    # ...existing code...
    features_df.to_csv('data/california_housing.csv', index=False)
    print("features saved as data/california_housing_preprocessed.csv")
except Exception as e:
    print(f"Error during preprocessing: {e}")
