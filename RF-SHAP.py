import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

# 读取数据
data = pd.read_excel(r'C:\Users\沐阳\Desktop\问卷1量表数据.xlsx')
# 设置中文字体为SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 提取特征和目标变量
X = data.iloc[:, 1:29]
y = data['FD30']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建随机森林模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 解释模型预测结果
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# 可视化SHAP值
plt.figure(figsize=(6, 4))
shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
#plt.title('SHAP Values for Features')
plt.show()

# 进行预测
y_pred = rf.predict(X_test)
plt.figure(figsize=(6, 4))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Samples')
plt.ylabel('Value')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()

# 评估模型
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('Mean Absolute Error:', mae)
print('Mean Absolute Percentage Error:', mape)
print('R^2 Score:', r2)

# SHAP模型预测结果可视化
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, X_test)
shap_values = explainer.shap_values(X_test)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, X_test)
shap.summary_plot(shap_values, X_test)
shap_interaction_values = explainer.shap_interaction_values(X_test)
shap.summary_plot(shap_interaction_values, X_test)






