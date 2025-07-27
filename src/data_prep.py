from sklearn.datasets import load_iris
import pandas as pd

# Load Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame
df.to_csv('data/iris.csv', index=False)
print("Saved iris.csv in data/")
