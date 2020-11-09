#generate recommendations for a list of unknown compounds
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor
from sklearn.svm import SVR

DE = pd.read_excel('combine_features_hv_temp_cif.xlsx')
array = DE.values
X = array[:,1:4049]
de = pd.read_excel('hv_temp_cif_labels.xlsx')
Y = de['hardness']
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, test_size=0.1,random_state=100, shuffle=True)
#train, test=train_test_split(array, train_size=0.9, test_size=0.1,random_state=100, shuffle=True)

#query by committee
SVM = SVR(C=100,gamma=0.01)
rf = RandomForestRegressor(max_depth=4, n_estimators=500)
gb = GradientBoostingRegressor(max_depth=4, learning_rate=0.18, n_estimators=500)
xgb = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree =0.7, learning_rate = 0.18, max_depth = 4, alpha = 5, n_estimators = 500, subsample=0.6)

committee=[rf, gb, xgb, SVM]

#predict_DE = pd.read_excel('PCD_66432_des_highT.xlsx')
#array = predict_DE.values
#pcd_predict_X = array[:,1:143]
#pcd_predict_X=scaler.transform(pcd_predict_X)

predict_y = np.zeros((len(y_test), 4))
i=0
for estimator in committee:
    estimator.fit(X_train, y_train)
    predict_y[:,i] = estimator.predict(X_test)
    i = i+1

pred_y = pd.DataFrame(predict_y)
std = pred_y.std(axis=1)
pred_y['std']=std

sorted_pred=pred_y.sort_values(by='std', ascending=False)

sorted_pred.head(10)

#adaptive max disagreement
n_repeat = 10
estimator=BaggingRegressor(base_estimator=xgb, bootstrap_features=True, max_features=100)

predict_y = np.zeros((len(y_test), n_repeat))

for i in range(n_repeat):
    estimator.fit(X_train, y_train)
    predict_y[:,i] = estimator.predict(X_test)
