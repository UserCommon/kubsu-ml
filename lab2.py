import pandas as pd

cnt = 2


def sep_print(func):
    def wrapper(*args, **kwargs):
        global cnt
        print("=" * 20)
        print(f"{cnt}>")
        cnt += 1
        func(*args, **kwargs)
        print("=" * 20)

    return wrapper


# 1.
df = pd.read_csv("./AirQualityUCI.csv", sep=";", decimal=",")


# 2.
@sep_print
def task_two():
    print("Первые 5 строк:")
    print(df[:5])

    print("Последние 5 строк:")
    print(df[-5:])


task_two()


# 3.
@sep_print
def task_three():
    rows, cols = df.shape
    print(f"Dimension: {rows}")
    print(f"Facts: {cols}")


task_three()


# 4.
@sep_print
def task_four():
    print("Names:")
    print(*df.columns)


task_four()


# 5.
@sep_print
def task_five():
    print("Missing values:")
    print(df.isnull().sum())


task_five()


# 6.
@sep_print
def task_six():
    print("Description:")
    print(df.describe())


task_six()


# 7.
@sep_print
def task_seven():
    print("Info:")
    print(df.info())


task_seven()


print(df)


# 8.
@sep_print
def task_eight():
    print("Temperatures:")
    print(f"unique: {df['T'].nunique()}")
    print(f"freq: {df['T'].value_counts()}")


task_eight()


# 9.
@sep_print
def task_nine():
    print("CO(GT) > 3 mg/m^3")
    print(df[(df["CO(GT)"] > 3) & (df["T"] < 20)])


task_nine()


# 10.
@sep_print
def task_ten():
    print("Add new column:")
    df["NOx_CO_ratio"] = df["NOx(GT)"] / df["CO(GT)"]
    print(df)


task_ten()


# 11.
@sep_print
def task_eleven():
    print("new size:")
    print(*df.shape)


task_eleven()


# 12.
@sep_print
def task_twelve():
    print("Most frequent value of T")
    print(df["T"].mode()[0])


task_twelve()


# 13
@sep_print
def task_thirteen():
    print("No C6H6(GT)")
    print(df[(df["CO(GT)"].isnull()) | (df["C6H6(GT)"] == -200)])


task_thirteen()


# 14
@sep_print
def task_fourteen():
    print("Minimal concentration of CO GT with T > 25")
    mask = df["T"] > 25
    print(df[mask]["CO(GT)"].min())
    print(*df[mask].shape)


task_fourteen()


# 15
@sep_print
def task_fifteen():
    print("Dimensions in which RH > 90")
    condition = df["RH"] > 90
    count_high_rh = condition.sum()
    high_rh_data = df[condition]
    print(f"Количество измерений с влажностью > 90%: {count_high_rh}")
    print("\nПодробная информация об этих измерениях (первые строки):")
    print(high_rh_data.head())
    print(df[condition].shape)


task_fifteen()


# 16.
@sep_print
def task_sixteen():
    print("Difference in mean temperature between T > 20 and T < 20")
    condition_greater_than_20 = df["T"] > 20
    condition_less_than_20 = df["T"] < 20
    print(
        round(abs(condition_greater_than_20.mean() - condition_less_than_20.mean()), 2)
    )


task_sixteen()


# 17.
@sep_print
def task_seventeen():
    print(
        """
        Создайте признак "High_Ozone", который будет равен 1, если уровень концентрации
        озона (PT08.S5(O3)) превышает среднее значение, и 0 в противном случае
        """
    )
    m = df["PT08.S5(O3)"].mean()
    df["High_Ozone"] = (df["PT08.S5(O3)"] > m).astype(int)
    print(df["High_Ozone"].value_counts())
    print(df.head())


task_seventeen()


# 18.
@sep_print
def task_eighteen():
    print("""
        Выведите наиболее распространенный уровень концентрации оксидов азота
        (NOx(GT))
    """)

    print(df["C6H6(GT)"].mode()[0])


task_eighteen()


# 19.
@sep_print
def task_nineteen():
    print("""
        Найдите количество измерений, в которых и уровень концентрации бензола
        (C6H6(GT)), и уровень концентрации оксидов азота (NOx(GT)) превышают свои средние
        значения.
    """)
    m_c6h6 = df["C6H6(GT)"].mean()
    m_nox = df["NOx(GT)"].mean()
    print(df[(df["C6H6(GT)"] > m_c6h6) & (df["NOx(GT)"] > m_nox)].shape[0])


task_nineteen()


# 20.
@sep_print
def task_twenty():
    print("""
        Найдите максимальную температуру (T) среди измерений, где уровень концентрации
        диоксида азота (NO2(GT)) ниже 50 мкг/м³.
    """)

    cond = df["NO2(GT)"] < 50
    print(df[cond]["T"].max())


task_twenty()


# 21.
@sep_print
def task_twentyone():
    global df
    print("""
        21. Найдите количество измерений, в которых уровень концентрации оксида углерода
        (CO(GT)) выше среднего значения по датасету. Выведите информацию об этих
        измерениях.
    """)

    co_col = df["CO(GT)"].replace(-200, pd.NA)
    mean_co = co_col.mean()

    filtered_df = df[df["CO(GT)"] > mean_co]

    count = filtered_df.shape[0]
    print(f"Среднее значение CO(GT): {mean_co:.2f}")
    print(f"Количество измерений выше среднего: {count}")

    print("\nИнформация об этих измерениях (первые 5):")
    print(filtered_df.head())

    nox_mode = df[df["NOx(GT)"] != -200]["NOx(GT)"].mode()
    if not nox_mode.empty:
        print(f"\nМода NOx(GT): {nox_mode[0]}")


task_twentyone()


# 22.
@sep_print
def task_twentytwo():
    print(
        """
        Разделите данные на две группы: измерения, сделанные при температуре выше
        среднего значения, и измерения, сделанные при температуре ниже среднего значения.
        Сравните средний уровень концентрации бензола (C6H6(GT)) в этих двух группах
        """
    )

    m = df["T"].mean()

    df1 = df[df["T"] >= m]
    df2 = df[df["T"] < m]

    a1 = df1["C6H6(GT)"].mean()
    a2 = df2["C6H6(GT)"].mean()

    print(f"""
        T1 > mean: {a1}
        T2 < mean: {a2}
        {"T1 > T2" if a1 > a2 else "T1 <= T2"}
    """)


task_twentytwo()
