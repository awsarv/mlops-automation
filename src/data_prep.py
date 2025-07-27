from sklearn.datasets import fetch_california_housing
import pandas as pd

housing = fetch_california_housing(as_frame=True)
df = housing.frame
df.to_csv('data/california_housing.csv', index=False)
print("California housing data saved as data/california_housing.csv")
