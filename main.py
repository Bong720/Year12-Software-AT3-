import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error


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

print("--- Training Level 1 AI ---")

# TODO 1: Create a variable X1 and set it to df_clean[['Maths_Advanced']].values
X1 = df_clean[['Maths_Advanced']].values
# TODO 2: Create a variable y and set it to df_clean['Software_Engineering_Final'].values
y = df_clean['Software_Engineering_Final'].values
# TODO 3: Use the train_test_split() function to divide X1 and y. Set test_size=0.2 and random_state=42.
X_train, X_test, y_train, y_test = train_test_split(
    X1, y, test_size=0.2, random_state=42
)
# TODO 4: Initialize a LinearRegression() model.
model = LinearRegression()
# TODO 5: .fit() your model using the training data only.
model.fit(X_train, y_train)
# TODO 6: Use your model to .predict() the outcomes for your test data.
y_pred = model.predict(X_test)
# TODO 7: Calculate the RMSE (Root Mean Squared Error) by comparing your predictions against the real y_test values. Print the result.
rmse = root_mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)

print("--- Extracting the Formula ---")

# TODO 1: Extract the weight using your_model.coef_[0] and the bias using your_model.intercept_
weight = model.coef_[0]
bias = model.intercept_
# TODO 2: Print out the final formula using those two numbers.
print(f"Formula: y = {weight:.2f}x + {bias:.2f}")
# TODO 3: Use plt.scatter() to plot the X1_test data against the y_test data. Make the dots blue.
plt.scatter(X_test, y_test, color="blue")
# TODO 4: Use plt.plot() to draw a line mapping X1_test against your AI's predictions. Make the line red.
y_predictions = model.predict(X_test)
plt.plot(X_test, y_predictions, color="red")
# TODO 5: Add a title, xlabel, ylabel, and plt.show() to display the graph.
plt.title("Linear Regression Predictions")
plt.xlabel("X1_test")
plt.ylabel("y_test")
plt.show()