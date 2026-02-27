import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Установка стиля для графиков
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

df = pd.read_csv("./AirQualityUCI.csv", sep=";", decimal=",", na_values=-200)

# Удаляем последние пустые столбцы
df = df.iloc[:, :-2]

# Удаляем строки, где все значения NaN
df = df.dropna(how="all")

# Заменяем пропущенные значения на NaN
df = df.replace(-200, np.nan)

print("\nПервые 5 строк:")
print(df.head())

print("\nИнформация о пропущенных значениях:")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\nОсновные статистики:")
print(df[["CO(GT)", "T", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "RH"]].describe())

# Для работы с данными заполним пропущенные значения медианой
for col in df.columns:
    if df[col].dtype in ["float64", "int64"]:
        df[col].fillna(df[col].median(), inplace=True)

# ============================================================================
# ЗАДАЧА 1: Столбчатая диаграмма распределения CO(GT)
# ============================================================================
print("\n" + "=" * 80)
print("ЗАДАЧА 1: Столбчатая диаграмма распределения CO(GT)")
print("=" * 80)

# Удаляем NaN значения для CO(GT)
co_data = df["CO(GT)"].dropna()

# Разбиваем CO(GT) на 8 интервалов
co_bins = pd.cut(co_data, bins=8)
co_counts = co_bins.value_counts().sort_index()

plt.figure(figsize=(12, 6))
co_counts.plot(kind="bar", color="steelblue", edgecolor="black", alpha=0.7)
plt.title(
    "Распределение концентрации оксида углерода CO(GT)", fontsize=14, fontweight="bold"
)
plt.xlabel("Диапазон концентрации CO (мг/м³)", fontsize=12)
plt.ylabel("Количество наблюдений", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("task1_co_distribution.png", dpi=300, bbox_inches="tight")
print("График сохранен: task1_co_distribution.png")

print("\nРаспределение по диапазонам:")
print(co_counts)
plt.close()

# ============================================================================
# ЗАДАЧА 2: Столбчатая диаграмма с логарифмическим масштабом
# ============================================================================
print("\n" + "=" * 80)
print("ЗАДАЧА 2: Столбчатая диаграмма с логарифмическим масштабом")
print("=" * 80)

plt.figure(figsize=(12, 6))
co_counts.plot(kind="bar", color="coral", edgecolor="black", alpha=0.7, logy=True)
plt.title(
    "Распределение концентрации CO(GT) (логарифмический масштаб)",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Диапазон концентрации CO (мг/м³)", fontsize=12)
plt.ylabel("Количество наблюдений (log scale)", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("task2_co_distribution_log.png", dpi=300, bbox_inches="tight")
print("График сохранен: task2_co_distribution_log.png")
plt.close()

# ============================================================================
# ЗАДАЧА 3: Две гистограммы температуры (CO выше и ниже среднего)
# ============================================================================
print("\n" + "=" * 80)
print("ЗАДАЧА 3: Гистограммы температуры для CO выше и ниже среднего")
print("=" * 80)

co_mean = df["CO(GT)"].mean()
temp_high_co = df[df["CO(GT)"] > co_mean]["T"]
temp_low_co = df[df["CO(GT)"] <= co_mean]["T"]

plt.figure(figsize=(12, 6))
plt.hist(
    temp_high_co,
    bins=20,
    color="red",
    edgecolor="black",
    alpha=0.6,
    label="CO выше среднего",
)
plt.hist(
    temp_low_co,
    bins=20,
    color="blue",
    edgecolor="black",
    alpha=0.6,
    label="CO ниже среднего",
)
plt.title(
    "Распределение температуры для разных уровней CO", fontsize=14, fontweight="bold"
)
plt.xlabel("Температура (°C)", fontsize=12)
plt.ylabel("Частота", fontsize=12)
plt.legend(fontsize=11)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("task3_temp_histograms.png", dpi=300, bbox_inches="tight")
print("График сохранен: task3_temp_histograms.png")
print(f"Среднее значение CO(GT): {co_mean:.2f} мг/м³")
print(f"Наблюдений с CO > среднего: {len(temp_high_co)}")
print(f"Наблюдений с CO ≤ среднего: {len(temp_low_co)}")
plt.close()

# ============================================================================
# ЗАДАЧИ 4-5: Гистограммы с плотностью распределения и легендой
# ============================================================================
print("\n" + "=" * 80)
print("ЗАДАЧИ 4-5: Гистограммы с плотностью распределения")
print("=" * 80)

plt.figure(figsize=(12, 6))
plt.hist(
    temp_high_co,
    bins=20,
    density=True,
    color="red",
    edgecolor="black",
    alpha=0.5,
    label="CO выше среднего",
)
plt.hist(
    temp_low_co,
    bins=20,
    density=True,
    color="blue",
    edgecolor="black",
    alpha=0.5,
    label="CO ниже среднего",
)
plt.title(
    "Плотность распределения температуры для разных уровней CO",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Температура (°C)", fontsize=12)
plt.ylabel("Плотность распределения", fontsize=12)
plt.legend(fontsize=11, loc="upper right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("task4_5_temp_density.png", dpi=300, bbox_inches="tight")
print("График сохранен: task4_5_temp_density.png")
plt.close()

# ============================================================================
# ЗАДАЧА 6: Гистограмма распределения C6H6 по времени суток
# ============================================================================
print("\n" + "=" * 80)
print("ЗАДАЧА 6: Распределение C6H6(GT) по времени суток")
print("=" * 80)

# Преобразуем время в правильный формат
df["Time_clean"] = df["Time"].str.replace(".", ":")
df["Hour"] = pd.to_datetime(
    df["Time_clean"], format="%H:%M:%S", errors="coerce"
).dt.hour


# Определяем время суток
def get_time_of_day(hour):
    if pd.isna(hour):
        return np.nan
    if 6 <= hour < 12:
        return "Утро (6-12)"
    elif 12 <= hour < 18:
        return "День (12-18)"
    elif 18 <= hour < 24:
        return "Вечер (18-24)"
    else:
        return "Ночь (0-6)"


df["TimeOfDay"] = df["Hour"].apply(get_time_of_day)

# Создаем гистограммы для каждого времени суток
plt.figure(figsize=(14, 7))
colors = {
    "Утро (6-12)": "gold",
    "День (12-18)": "orange",
    "Вечер (18-24)": "crimson",
    "Ночь (0-6)": "navy",
}

for time_period in ["Ночь (0-6)", "Утро (6-12)", "День (12-18)", "Вечер (18-24)"]:
    data = df[df["TimeOfDay"] == time_period]["C6H6(GT)"].dropna()
    if len(data) > 0:
        plt.hist(
            data,
            bins=20,
            density=True,
            alpha=0.5,
            label=time_period,
            color=colors[time_period],
            edgecolor="black",
        )

plt.title(
    "Плотность распределения концентрации бензола (C6H6) по времени суток",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Концентрация C6H6 (мкг/м³)", fontsize=12)
plt.ylabel("Плотность распределения", fontsize=12)
plt.legend(fontsize=11)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("task6_c6h6_by_time.png", dpi=300, bbox_inches="tight")
print("График сохранен: task6_c6h6_by_time.png")

print("\nСредние концентрации C6H6 по времени суток:")
for period in ["Ночь (0-6)", "Утро (6-12)", "День (12-18)", "Вечер (18-24)"]:
    mean_val = df[df["TimeOfDay"] == period]["C6H6(GT)"].mean()
    count = df[df["TimeOfDay"] == period]["C6H6(GT)"].count()
    print(f"  {period}: {mean_val:.2f} мкг/м³ (n={count})")
plt.close()

# ============================================================================
# ЗАДАЧА 7: Boxplots для CO(GT) по времени суток
# ============================================================================
print("\n" + "=" * 80)
print("ЗАДАЧА 7: Boxplots для CO(GT) по времени суток")
print("=" * 80)

time_order = ["Ночь (0-6)", "Утро (6-12)", "День (12-18)", "Вечер (18-24)"]
data_for_boxplot = [
    df[df["TimeOfDay"] == period]["CO(GT)"].dropna().values for period in time_order
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Boxplots
bp = ax1.boxplot(
    data_for_boxplot,
    labels=time_order,
    patch_artist=True,
    boxprops=dict(facecolor="lightblue", alpha=0.7),
    medianprops=dict(color="red", linewidth=2),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
)

ax1.set_title(
    "Boxplot: Распределение CO(GT) по времени суток", fontsize=13, fontweight="bold"
)
ax1.set_xlabel("Время суток", fontsize=11)
ax1.set_ylabel("Концентрация CO (мг/м³)", fontsize=11)
ax1.grid(axis="y", alpha=0.3)
ax1.tick_params(axis="x", rotation=15)

# Гистограммы
for i, (period, color) in enumerate(
    zip(time_order, ["navy", "gold", "orange", "crimson"])
):
    data = df[df["TimeOfDay"] == period]["CO(GT)"].dropna()
    if len(data) > 0:
        ax2.hist(data, bins=15, alpha=0.5, label=period, color=color, edgecolor="black")

ax2.set_title(
    "Гистограммы: Распределение CO(GT) по времени суток", fontsize=13, fontweight="bold"
)
ax2.set_xlabel("Концентрация CO (мг/м³)", fontsize=11)
ax2.set_ylabel("Частота", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("task7_co_by_time_boxplot.png", dpi=300, bbox_inches="tight")
print("График сохранен: task7_co_by_time_boxplot.png")

print("\nСтатистика CO(GT) по времени суток:")
for period in time_order:
    data = df[df["TimeOfDay"] == period]["CO(GT)"].dropna()
    if len(data) > 0:
        print(f"\n{period}:")
        print(f"  Среднее: {data.mean():.2f} мг/м³")
        print(f"  Медиана: {data.median():.2f} мг/м³")
        print(f"  Std: {data.std():.2f} мг/м³")
        print(f"  Количество: {len(data)}")
plt.close()

# ============================================================================
# ЗАДАЧА 8: Зависимость загрязняющих веществ от температуры
# ============================================================================
print("\n" + "=" * 80)
print("ЗАДАЧА 8: Зависимость загрязнителей от температуры")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

pollutants = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
colors_scatter = ["red", "green", "blue", "purple"]
titles = [
    "Зависимость CO(GT) от температуры",
    "Зависимость C6H6(GT) от температуры",
    "Зависимость NOx(GT) от температуры",
    "Зависимость NO2(GT) от температуры",
]

for idx, (ax, pollutant, color, title) in enumerate(
    zip(axes.flat, pollutants, colors_scatter, titles)
):
    # Убираем NaN значения
    mask = df["T"].notna() & df[pollutant].notna()
    temp_clean = df.loc[mask, "T"]
    poll_clean = df.loc[mask, pollutant]

    # Scatter plot
    ax.scatter(temp_clean, poll_clean, alpha=0.3, s=10, color=color)

    # Линия тренда (полином 2-й степени)
    if len(temp_clean) > 3:
        z = np.polyfit(temp_clean, poll_clean, 2)
        p = np.poly1d(z)
        temp_sorted = np.sort(temp_clean)
        ax.plot(
            temp_sorted,
            p(temp_sorted),
            "r--",
            linewidth=2,
            label="Тренд (полином 2-й степени)",
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Температура (°C)", fontsize=10)
    ax.set_ylabel(f"{pollutant}", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("task8_pollutants_vs_temp.png", dpi=300, bbox_inches="tight")
print("График сохранен: task8_pollutants_vs_temp.png")

# Вычисляем корреляции
print("\nКорреляция загрязнителей с температурой:")
for pollutant in pollutants:
    mask = df["T"].notna() & df[pollutant].notna()
    if mask.sum() > 0:
        corr = df.loc[mask, [pollutant, "T"]].corr().iloc[0, 1]
        print(f"  {pollutant} vs T: {corr:.3f}")
plt.close()

# Гистограммы по температурным диапазонам
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, (ax, pollutant, color) in enumerate(
    zip(axes.flat, pollutants, colors_scatter)
):
    # Разбиваем температуру на 5 диапазонов
    temp_bins = pd.cut(
        df["T"], bins=5, labels=["Холодно", "Прохладно", "Умеренно", "Тепло", "Жарко"]
    )

    for temp_range in temp_bins.cat.categories:
        data = df[temp_bins == temp_range][pollutant].dropna()
        if len(data) > 0:
            ax.hist(data, bins=15, alpha=0.5, label=temp_range, edgecolor="black")

    ax.set_title(
        f"Распределение {pollutant} при разных температурах",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel(f"{pollutant}", fontsize=10)
    ax.set_ylabel("Частота", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    "task8_pollutants_hist_by_temp.png",
    dpi=300,
    bbox_inches="tight",
)
print("График сохранен: task8_pollutants_hist_by_temp.png")
plt.close()

# ============================================================================
# ЗАДАЧА 9*: Area plot - изменение концентраций во времени
# ============================================================================
print("\n" + "=" * 80)
print("ЗАДАЧА 9*: Area plot - изменение концентраций во времени")
print("=" * 80)

# Преобразуем дату в datetime
df["Date_parsed"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")

# Агрегируем данные по дням
df_daily = (
    df.groupby(df["Date_parsed"].dt.date)
    .agg({"CO(GT)": "mean", "C6H6(GT)": "mean", "NOx(GT)": "mean", "NO2(GT)": "mean"})
    .reset_index()
)

# Удаляем NaN
df_daily = df_daily.dropna()

# Берем первые 90 дней для визуализации
df_plot = df_daily.head(90).copy()

if len(df_plot) > 0:
    # Нормализуем данные для лучшего отображения
    pollutants_normalized = df_plot[["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]].copy()
    pollutants_normalized["CO(GT)"] = pollutants_normalized["CO(GT)"] * 50
    pollutants_normalized["C6H6(GT)"] = pollutants_normalized["C6H6(GT)"] * 10
    pollutants_normalized["NOx(GT)"] = pollutants_normalized["NOx(GT)"] / 2
    pollutants_normalized["NO2(GT)"] = pollutants_normalized["NO2(GT)"] / 1

    plt.figure(figsize=(16, 8))
    plt.stackplot(
        range(len(df_plot)),
        pollutants_normalized["CO(GT)"],
        pollutants_normalized["C6H6(GT)"],
        pollutants_normalized["NOx(GT)"],
        pollutants_normalized["NO2(GT)"],
        labels=["CO(GT) × 50", "C6H6(GT) × 10", "NOx(GT) / 2", "NO2(GT)"],
        colors=["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"],
        alpha=0.8,
    )

    plt.title(
        "Изменение концентраций загрязняющих веществ во времени (первые 90 дней)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Дни наблюдений", fontsize=12)
    plt.ylabel("Нормализованные концентрации (условные единицы)", fontsize=12)
    plt.legend(loc="upper left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        "task9_area_plot_pollutants.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("График сохранен: task9_area_plot_pollutants.png")
    plt.close()

    # График с двумя осями
    fig, ax1 = plt.subplots(figsize=(16, 8))

    color1 = "tab:red"
    ax1.set_xlabel("Дни наблюдений", fontsize=12)
    ax1.set_ylabel("CO(GT) и C6H6(GT)", color=color1, fontsize=12)
    ax1.fill_between(
        range(len(df_plot)), df_plot["CO(GT)"], alpha=0.5, color="red", label="CO(GT)"
    )
    ax1.fill_between(
        range(len(df_plot)),
        df_plot["C6H6(GT)"],
        alpha=0.5,
        color="orange",
        label="C6H6(GT)",
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "tab:blue"
    ax2.set_ylabel("NOx(GT) и NO2(GT)", color=color2, fontsize=12)
    ax2.fill_between(
        range(len(df_plot)),
        df_plot["NOx(GT)"],
        alpha=0.5,
        color="blue",
        label="NOx(GT)",
    )
    ax2.fill_between(
        range(len(df_plot)),
        df_plot["NO2(GT)"],
        alpha=0.5,
        color="cyan",
        label="NO2(GT)",
    )
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.legend(loc="upper right", fontsize=10)

    plt.title(
        "Динамика концентраций загрязняющих веществ (две оси)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        "task9_area_plot_dual_axis.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("График сохранен: task9_area_plot_dual_axis.png")
    plt.close()

# ============================================================================
# ИТОГОВАЯ ИНФОРМАЦИЯ
# ============================================================================
print("1. task1_co_distribution.png - Столбчатая диаграмма распределения CO(GT)")
print("2. task2_co_distribution_log.png - То же с логарифмическим масштабом")
print("3. task3_temp_histograms.png - Гистограммы температуры для разных уровней CO")
print("4. task4_5_temp_density.png - Плотность распределения температуры")
print("5. task6_c6h6_by_time.png - Распределение C6H6 по времени суток")
print("6. task7_co_by_time_boxplot.png - Boxplot и гистограммы CO по времени суток")
print("7. task8_pollutants_vs_temp.png - Scatter plots зависимости от температуры")
print("8. task8_pollutants_hist_by_temp.png - Гистограммы при разных температурах")
print("9. task9_area_plot_pollutants.png - Area plot изменения концентраций")
print("10. task9_area_plot_dual_axis.png - Area plot с двумя осями")
print("\nВсе графики сохранены в высоком разрешении (300 DPI)")

# Итоговая статистика
print("\n" + "=" * 80)
print("СТАТИСТИЧЕСКИЙ АНАЛИЗ РЕАЛЬНОГО ДАТАСЕТА")
print("=" * 80)

print("\n1. Информация о датасете:")
print(f"   Всего записей: {len(df)}")
print(f"   Период наблюдений: {df['Date'].min()} - {df['Date'].max()}")

print("\n2. Статистика CO(GT):")
co_stats = df["CO(GT)"].describe()
print(f"   Среднее: {co_stats['mean']:.2f} мг/м³")
print(f"   Медиана: {co_stats['50%']:.2f} мг/м³")
print(f"   Std: {co_stats['std']:.2f} мг/м³")
print(f"   Мин: {co_stats['min']:.2f} мг/м³")
print(f"   Макс: {co_stats['max']:.2f} мг/м³")

print("\n3. Средние значения по времени суток:")
for period in time_order:
    co_mean_period = df[df["TimeOfDay"] == period]["CO(GT)"].mean()
    print(f"   {period}: {co_mean_period:.2f} мг/м³")

print("\n" + "=" * 80)
print("Все задачи лабораторной работы выполнены успешно!")
print("=" * 80)
