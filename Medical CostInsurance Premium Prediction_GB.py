import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the Insurance Dataset from a reliable online public source
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
df = pd.read_csv(url)

print("First 5 rows of Insurance Data:")
print(df.head())

# Define Features and Target (predicting medical 'charges')
X = df.drop('charges', axis=1)
y = df['charges']

# 2. Preprocessing: Encode Categorical columns (sex, smoker, region)
categorical_cols = ['sex', 'smoker', 'region']
numerical_cols = ['age', 'bmi', 'children']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols) # drop='first' avoids multi-collinearity
    ])

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing transformations
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 4. Train the Gradient Boosting Regressor
gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_regressor.fit(X_train_processed, y_train)

# 5. Predict and Evaluate
y_pred = gb_regressor.predict(X_test_processed)

print("\n--- Regression Metrics ---")
print(f"Mean Absolute Error (MAE): ${mean_absolute_error(y_test, y_pred):.2f}")
print(f"Root Mean Squared Error (RMSE): ${mean_squared_error(y_test, y_pred, squared=False):.2f}")
print(f"R-squared (R2) Score: {r2_score(y_test, y_pred):.4f}")