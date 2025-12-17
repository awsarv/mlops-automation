"""
Fraud Detection Data Preparation
Generates synthetic credit card transaction data for fraud detection demo.
In production, this would load real transaction data from a secure source.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# Configuration
N_SAMPLES = 50000
FRAUD_RATIO = 0.02  # 2% fraud rate (realistic)

n_fraud = int(N_SAMPLES * FRAUD_RATIO)
n_legitimate = N_SAMPLES - n_fraud

print(f"Generating {N_SAMPLES} transactions ({n_fraud} fraud, {n_legitimate} legitimate)")

# Generate legitimate transactions
legitimate = pd.DataFrame({
    'amount': np.random.exponential(scale=50, size=n_legitimate).clip(1, 5000),
    'hour': np.random.randint(6, 23, size=n_legitimate),  # Mostly daytime
    'day_of_week': np.random.choice(range(7), size=n_legitimate),
    'merchant_category': np.random.choice(range(15), size=n_legitimate),
    'distance_from_home': np.random.exponential(scale=10, size=n_legitimate).clip(0, 100),
    'distance_from_last_transaction': np.random.exponential(scale=5, size=n_legitimate).clip(0, 50),
    'ratio_to_median_purchase': np.random.normal(1.0, 0.3, size=n_legitimate).clip(0.1, 3),
    'repeat_retailer': np.random.choice([0, 1], size=n_legitimate, p=[0.3, 0.7]),
    'used_chip': np.random.choice([0, 1], size=n_legitimate, p=[0.2, 0.8]),
    'used_pin': np.random.choice([0, 1], size=n_legitimate, p=[0.4, 0.6]),
    'online_order': np.random.choice([0, 1], size=n_legitimate, p=[0.7, 0.3]),
    'is_fraud': 0
})

# Generate fraudulent transactions (different patterns)
fraud = pd.DataFrame({
    'amount': np.random.exponential(scale=500, size=n_fraud).clip(100, 10000),
    'hour': np.random.randint(0, 6, size=n_fraud),  # More fraud at night
    'day_of_week': np.random.choice(range(7), size=n_fraud),
    'merchant_category': np.random.randint(0, 15, size=n_fraud),
    'distance_from_home': np.random.exponential(scale=50, size=n_fraud).clip(10, 500),
    'distance_from_last_transaction': np.random.exponential(scale=30, size=n_fraud).clip(5, 200),
    'ratio_to_median_purchase': np.random.normal(3.0, 1.5, size=n_fraud).clip(1.5, 10),
    'repeat_retailer': np.random.choice([0, 1], size=n_fraud, p=[0.8, 0.2]),
    'used_chip': np.random.choice([0, 1], size=n_fraud, p=[0.7, 0.3]),
    'used_pin': np.random.choice([0, 1], size=n_fraud, p=[0.85, 0.15]),
    'online_order': np.random.choice([0, 1], size=n_fraud, p=[0.3, 0.7]),
    'is_fraud': 1
})

# Combine and shuffle
df = pd.concat([legitimate, fraud], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Create data directory if not exists
os.makedirs('data', exist_ok=True)

# Save dataset
df.to_csv('data/credit_card_transactions.csv', index=False)

print(f"Dataset saved to data/credit_card_transactions.csv")
print(f"Shape: {df.shape}")
print(f"Fraud distribution:\n{df['is_fraud'].value_counts()}")
print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
