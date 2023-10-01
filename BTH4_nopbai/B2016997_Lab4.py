# cau 1

from sklearn.metrics import mean_squared_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X = [150, 147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]
Y = [90, 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]

plt.scatter(X, Y)
plt.xlabel('Chiều cao (cm)')
plt.ylabel('Cân nặng (kg)')
plt.title('Biểu đồ phân bố chiều cao và cân nặng')
plt.show()


X = [147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]
Y = [49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]

# Tính hệ số a, b của đường hồi quy
a, b = np.polyfit(X, Y, 1)

# Vẽ scatter plot
plt.scatter(X, Y)

# Vẽ đường hồi quy
plt.plot(X, a*np.array(X) + b, color='red')

# Đặt tên trục và tiêu đề
plt.xlabel('Chiều cao (cm)')
plt.ylabel('Cân nặng (kg)')
plt.title('Biểu đồ phân bố chiều cao và cân nặng với đường hồi quy')

plt.show()


# cau 2


# Đọc dữ liệu từ file csv vào dataframe pandas
data = pd.read_csv('Housing_2019.csv')

# Tách dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    data[['lotsize', 'bedrooms', 'stories', 'garagepl']], data['price'], test_size=0.2, random_state=42)

# Huấn luyện mô hình trên tập huấn luyện
model = LinearRegression()
model.fit(X_train, y_train)

# Dự báo giá nhà trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình sử dụng MSE và RMSE trên tập kiểm tra
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print('MSE:', mse)
print('RMSE:', rmse)
