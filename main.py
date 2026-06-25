import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.neural_network import MLPRegressor

print("--- Loading and Cleaning Data ---")
df_raw = pd.read_csv("master_markbook.csv", encoding='latin-1')
print(len(df_raw))
df_clean = df_raw.dropna()
df_clean = df_clean[
    df_clean['Maths_Advanced'].between(0, 100) &
    df_clean['Physics'].between(0, 100) &
    df_clean['Software_Engineering_Final'].between(0, 100)
]
print(len(df_clean))

print("--- Training Level 1 AI ---")
X1 = df_clean[['Maths_Advanced']].values
y = df_clean['Software_Engineering_Final'].values
X1_train, X1_test, y_train, y_test = train_test_split(
    X1, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X1_train, y_train)
y_pred = model.predict(X1_test)
rmse = root_mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)

print("--- Extracting the Formula ---")
weight = model.coef_[0]
bias = model.intercept_
print(f"Formula: y = {weight:.2f}x + {bias:.2f}")
plt.scatter(X1_test, y_test, color="blue")
y_predictions = model.predict(X1_test)
plt.plot(X1_test, y_predictions, color="red")
plt.title("Linear Regression Predictions")
plt.xlabel("X1_test")
plt.ylabel("y_test")
plt.show()

print("--- Training OOP Engine ---")
class MarkPredictor:
    def __init__(self):
        self.model = LinearRegression()
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
my_ai = MarkPredictor()
my_ai.fit(X1_train, y_train)
y_predictions = my_ai.predict(X1_test)
rmse = np.sqrt(mean_squared_error(y_test, y_predictions))
print("RMSE:", rmse)

print("--- Training Level 2 AI (Multi-Subject) ---")
X2 = df_clean[['Maths_Advanced', 'Physics']].values
X2_train, X2_test, y_train, y_test = train_test_split(
    X2, y, test_size=0.2, random_state=42
)
scaler_level2 = StandardScaler()
X2_train_scaled = scaler_level2.fit_transform(X2_train)
X2_test_scaled = scaler_level2.transform(X2_test)
my_ai_level2 = MarkPredictor()
my_ai_level2.fit(X2_train_scaled, y_train)
y_predictions = my_ai_level2.predict(X2_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_predictions))
print("Level 2 RMSE:", rmse)


print("--- Running Bias Audit (80% Rule) ---")
group_a = df_clean[df_clean['Physics'] > 70]
group_b = df_clean[df_clean['Modern_History'] > 70]
pass_rate_a = (group_a['Software_Engineering_Final'] >= 50).mean()
print("Group A Pass Rate:", pass_rate_a)
pass_rate_b = (group_b['Software_Engineering_Final'] >= 50).mean()
print("Group B Pass Rate:", pass_rate_b)
disparate_impact_ratio = pass_rate_b / pass_rate_a
print("Disparate Impact Ratio:", disparate_impact_ratio)
if disparate_impact_ratio < 0.8:
    print("WARNING: Potential bias detected.")
else:
    print("Audit passed: No bias detected under the 80% rule.")

print("--- Cross-Validation Check ---")
scaler = StandardScaler()
X2_scaled = scaler.fit_transform(X2)
cv_scores = cross_val_score(
    LinearRegression(),
    X2_scaled,
    y,
    cv=5,
    scoring='neg_root_mean_squared_error'
)
fold_rmses = -cv_scores

print("Fold 1 RMSE:", fold_rmses[0])
print("Fold 2 RMSE:", fold_rmses[1])
print("Fold 3 RMSE:", fold_rmses[2])
print("Fold 4 RMSE:", fold_rmses[3])
print("Fold 5 RMSE:", fold_rmses[4])

cv_rmse = -cv_scores.mean()
print("Cross-Validation RMSE:", cv_rmse)

print("--- Logic Gatekeeper ---")
def check_data_reliability(attendance_percentage):
    if attendance_percentage < 50.0:
        print("ACCESS DENIED")
        return False
    else:
        print("ACCESS GRANTED")
        return True
check_data_reliability(45)
check_data_reliability(92)

print("--- Extension: Neural Network Test ---")
nn_model = MLPRegressor(
    hidden_layer_sizes=(16, 8),
    max_iter=1500,
    random_state=42
)
nn_model.fit(X2_train_scaled, y_train)
nn_predictions = nn_model.predict(X2_test_scaled)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_predictions))
print("Level 2 Linear RMSE:", rmse)
print("Neutral Network RMSE:", nn_rmse)

print("--- Predict Missing Mark for New Student ---")
new_student = np.array([[80, 82]])
new_student_scaled = scaler_level2.transform(new_student)
predicted_mark = my_ai_level2.model.predict(new_student_scaled)
print("Predicted Physics Final Mark:", predicted_mark[0])