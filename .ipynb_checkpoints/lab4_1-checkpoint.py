#########
# SCIKIT LAB
########
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("./AirQualityUCI.csv", sep=";", decimal=",")
# Чистка данных
df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)
df.replace(-200, np.nan, inplace=True)
df = df.dropna(subset=["C6H6(GT)", "CO(GT)"])

## 1.
from sklearn.linear_model import LinearRegression

x_c6h6 = df[["C6H6(GT)"]]  # X
y_cogt = df["CO(GT)"]  # Y

x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(
    x_c6h6, y_cogt, test_size=0.2, random_state=42
)

linear_regression = LinearRegression()
linear_regression.fit(x_train_1, y_train_1)

cogt_pred = linear_regression.predict(x_test_1)

mse = mean_squared_error(y_test_1, cogt_pred)
r2 = r2_score(y_test_1, cogt_pred)

print(f"MSE: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

## Отрисовка
plt.switch_backend("Agg")

plt.figure(figsize=(10, 6))

# Рисуем реальные точки из тестовой выборки (серым цветом)
plt.scatter(x_test_1, y_test_1, color="gray", alpha=0.5, label="Реальные данные (тест)")

# Рисуем красную линию регрессии поверх точек
# Для линии берем отсортированные X, чтобы она была ровной
sorted_idx = x_test_1.squeeze().argsort()
plt.plot(
    x_test_1.squeeze().iloc[sorted_idx],
    cogt_pred[sorted_idx],
    color="red",
    linewidth=3,
    label="Модель линейной регрессии",
)

# Оформление графика
plt.title(f"Прогноз CO(GT) по C6H6(GT)\n$R^2 = {r2:.3f}$")
plt.xlabel("Концентрация Бензола (C6H6(GT))")
plt.ylabel("Концентрация Угарного газа (CO(GT))")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

# Сохраняем в файл, а не показываем на экране
plot_filename = "regression_plot.png"
plt.savefig(
    plot_filename, dpi=300, bbox_inches="tight"
)  # dpi=300 для хорошего качества

# Закрываем график, чтобы очистить память
plt.close()

print(f"График успешно сохранен в файл: {plot_filename}")
