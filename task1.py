import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import numpy as np

data = pd.read_csv(r'DPS920-Assignment2\house_price.csv')

X = data[["size", "bedroom"]]
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("LinearRegression Intercept:", lr_model.intercept_)
print("LinearRegression Size coefficient:", lr_model.coef_[0])
print("LinearRegression Bedroom coefficient:", lr_model.coef_[1])

mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)

print("LinearRegression MAE:", mae_lr)
print("LinearRegression MSE:", mse_lr)
print("LinearRegression RMSE:", rmse_lr)
print("LinearRegression MAPE:", mape_lr)

# SGDRegressor
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd_model.fit(X_train, y_train)
y_pred_sgd = sgd_model.predict(X_test)

print("SGDRegressor Intercept:", sgd_model.intercept_[0])
print("SGDRegressor Size coefficient:", sgd_model.coef_[0])
print("SGDRegressor Bedroom coefficient:", sgd_model.coef_[1])

mae_sgd = mean_absolute_error(y_test, y_pred_sgd)
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
rmse_sgd = np.sqrt(mse_sgd)
mape_sgd = mean_absolute_percentage_error(y_test, y_pred_sgd)

print("SGDRegressor MAE:", mae_sgd)
print("SGDRegressor MSE:", mse_sgd)
print("SGDRegressor RMSE:", rmse_sgd)
print("SGDRegressor MAPE:", mape_sgd)
