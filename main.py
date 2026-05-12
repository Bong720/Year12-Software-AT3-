import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler



print("--- Loading and Cleaning Data ---")

# TODO 1: Load the raw dataset into a variable called df_raw (Hint: use encoding='latin-1' to avoid Unicode errors)
df_raw = pd.read_csv("master_markbook.csv", encoding='latin-1')
# TODO 2: Print the length (number of rows) of df_raw so you know your starting number.
print(len(df_raw))
# TODO 3: Create a new variable called df_clean. Use the Pandas .dropna() method to delete any rows with missing data.
df_clean = df_raw.dropna()
# TODO 4: Update df_clean by filtering it. You must only keep rows where 'Maths_Advanced', 'Physics', and 'Software_Engineering_Final' are .between(0, 100).
df_clean = df_clean[
    df_clean['Maths_Advanced'].between(0, 100) &
    df_clean['Physics'].between(0, 100) &
    df_clean['Software_Engineering_Final'].between(0, 100)
]
# TODO 5: Print the length of df_clean. (You should have fewer rows than you started with!)
print(len(df_clean))