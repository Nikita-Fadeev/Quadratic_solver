# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC !pip install gurobipy
# MAGIC from datetime import datetime, timedelta
# MAGIC import pandas as pd, numpy as np
# MAGIC import matplotlib.pyplot as plt, seaborn as sns
# MAGIC
# MAGIC from IPython.display import display, clear_output
# MAGIC pd.set_option("display.max_rows", 300)
# MAGIC pd.set_option("display.max_columns", 200)
# MAGIC pd.set_option("display.max_colwidth", 200)
# MAGIC pd.set_option('display.float_format', '{:.4f}'.format)
# MAGIC clear_output()

# COMMAND ----------

# MAGIC %md
# MAGIC ### load data

# COMMAND ----------

# set population as in Raleigh market
POPULATION = {'AkronOH': 1_260_000}

# COMMAND ----------

# load schedule data
FILE = 'Pairwise Duplication 2025-06-02[77].xlsm'
SELECT = {'market':'market', 'station':'channel', 'dow':'dow', 'daypart':'daypart', 'P18+ Impressions':'aqh_cost'}
keys = ['channel', 'dow', 'daypart']
origin = (pd.read_excel(FILE)[list(SELECT.keys())]
            .rename(columns=SELECT)
            .drop_duplicates(keys)
          )

channels = origin['channel'].drop_duplicates().to_list()
dow = origin['dow'].drop_duplicates().to_list()
dayparts = origin['daypart'].drop_duplicates().to_list()
print(f'channels: {len(channels)}, dow: {len(dow)}, dayparts: {len(dayparts)}, all: {len(origin)}')

# COMMAND ----------

# load cume and aqh data
SELECT = {'Calls1':'channel', 'DOW1':'dow', 'Daypart':'daypart', 'Unrounded AQH':'aqh', 'Unrounded Cume':'cume'}
dow_map = {'MONDAY': 1, 'TUESDAY':2, 'WEDNESDAY':3, 'THURSDAY':4, 'FRIDAY':5, 'SATURDAY':6, 'SUNDAY':7}
cume_aqh = (pd.read_excel(FILE, sheet_name='Discrete')[list(SELECT.keys())]
              .rename(columns=SELECT)
              .assign(dow = lambda x: x['dow'].map(dow_map))
              .assign(aqh = lambda x: x['aqh'].astype(int))
              .assign(cume = lambda x: x['cume'].astype(int))
            )


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(cume_aqh['cume'], ax=axes[0]), sns.histplot(cume_aqh['aqh'], ax=axes[1])

# COMMAND ----------

# different channles names in origin and cume_aqh - replace channels names for simulation 
replace = { origin['channel'].unique()[index]: value for index, value in enumerate(cume_aqh['channel'].unique())}

# COMMAND ----------

# prepare data
pd.set_option('mode.use_inf_as_na', True)
ON = ['channel', 'dow', 'daypart']
data = (origin.assign(channel = lambda x: x['channel'].map(replace))
              .assign(obj = lambda x: x[['channel',  'dow', 'daypart']].astype(str).agg('_'.join, axis=1))
              .merge(cume_aqh, on=ON, how='left')
              .assign(cpm = lambda x: (x['aqh_cost'] / x['aqh']))
              .assign(cpm = lambda x:  x['cpm'].fillna(x['cpm'].mean()))
              .assign(population = lambda x: x['market'].map(POPULATION))
              .assign(reach = lambda x: x['cume'] / x['population']))

# COMMAND ----------

# MAGIC %md
# MAGIC ### build correlation table

# COMMAND ----------

# random correlation approach (not used)
np.random.seed(seed=42)
SELECT = ['market', 'obj']
corr_table = (data[SELECT].merge(data[SELECT], on='market', how='left')
             .loc[lambda x: ~(x['obj_x'] == x['obj_y'])]
             .assign(corr = lambda x: np.random.uniform(0.5, 0.7, len(x)))
)
corr_table

# COMMAND ----------

# make a expert assumption by correaltion
np.random.seed(seed=42)
BASE_CORR = 0.05
SELECT = ['market', 'obj', 'channel', 'dow', 'daypart']
corr_table = (data[SELECT].merge(data[SELECT], on='market', how='left')
             .loc[lambda x: ~(x['obj_x'] == x['obj_y'])]
             .assign(channel_corr = lambda x: np.where(x['channel_x'] == x['channel_y'], 1, 0))
             .assign(dow_corr = lambda x: np.where(x['dow_x'] == x['dow_y'], 1, 0))
             .assign(daypart_corr = lambda x: np.where(x['daypart_x'] == x['daypart_y'], 1, 0))
             .assign(corr = lambda x: BASE_CORR + x['channel_corr'] * 0.3 + x['dow_corr']* 0.2 + x['daypart_corr'] * 0.1)
             .drop(['channel_corr', 'dow_corr', 'daypart_corr', 'channel_x', 'channel_y', 'dow_x', 'dow_y', 'daypart_x', 'daypart_y'], axis=1)
)
corr_table

# COMMAND ----------

# take subset of data
data_len = len(data)
SHARE = 0.7
allow_matrix = np.random.uniform(0, 1, size=(data_len))  > SHARE

# COMMAND ----------

# build correlation matrix
# obj = channlel_dow_daypart
KEY = 'obj'
JOIN = 'market'
SELECT = [JOIN , KEY]
schedule = data[allow_matrix]
correlation_matrix = (schedule[SELECT].merge(schedule[SELECT], on=JOIN, how='left')
                                      .merge(corr_table, on=[KEY+'_x', KEY+'_y', 'market'], how='left')
                                      .assign(corr = lambda x: x['corr'].fillna(0))
                                      .pivot_table(index=KEY+'_x', columns=KEY+'_y', values='corr', fill_value=0, aggfunc='sum')
                      )
order_index = correlation_matrix.index

# COMMAND ----------

schedule

# COMMAND ----------

# correlation matrix must be symmetric
def is_symmetric(matrix, tol=1e-8):
    return np.allclose(matrix, matrix.T, atol=tol)

if not is_symmetric(correlation_matrix): 
    print("make symmetric")
    make_symmetric = lambda x: x + x.T - np.diag(np.diag(x))
    correlation_matrix = make_symmetric(correlation_matrix)

# COMMAND ----------

SELECT = ['reach']

lambdas = (schedule.set_index('obj').reindex(order_index)
                   [SELECT].to_numpy().flatten())
corr = correlation_matrix.to_numpy()

# COMMAND ----------

lambdas.shape, corr.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gurobi model - schedule optimisation. Poisson model - combined reach estimation 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Input

# COMMAND ----------

import gurobipy as gp

# reach (λ) 
lambdas = schedule[['reach']].to_numpy().flatten()

# Amount of posts
k = len(lambdas)

# listen duration (not used)
duration = 4 * 60
T = (schedule['aqh'] * duration / (schedule['cume'] )) * 60
T = T.fillna(T.mean()).to_numpy().flatten()

# Publication cost for each channel-dow-daypart
cost = schedule[['aqh_cost']].to_numpy().flatten()

# correlation matrix
C = corr

# Budget
budget = 20_000  
max_posts = 3

# COMMAND ----------

# show input data
print("reach:", lambdas[:10], f"shape {lambdas.shape}")
print('cost:', cost[:10], f"shape {cost.shape}")
print('Correlation matrix:', C[:5,:5], f"shape {C.shape}")
print('budget:', budget)
print('max_posts:', max_posts)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Poisson reach estimation

# COMMAND ----------

# check lambdas and corr constrains
def do_assert(lambdas, corr):
    """
    Проверяет корректность параметров lambdas и корреляционной матрицы corr.
    
    Параметры:
    ----------
    lambdas : array-like
        Массив интенсивностей для каждого канала (должны быть положительными)
    corr : 2D array-like
        Корреляционная матрица (должна быть симметричной, с 1 на диагонали,
        и положительно полуопределённой)
        
    Выбрасывает:
    -----------
    AssertionError
        Если параметры не проходят проверки
    """
    # Преобразуем в numpy массивы
    lambdas = np.asarray(lambdas)
    corr = np.asarray(corr)
    
    # Проверки для lambdas
    assert len(lambdas.shape) == 1, "lambdas должен быть 1D массивом"
    assert np.all(lambdas >= 0), "Все значения lambdas должны быть > 0"
    assert not np.any(np.isnan(lambdas)), "lambdas не должен содержать NaN"
    
    # Проверки для corr
    assert len(corr.shape) == 2, "corr должна быть 2D матрицей"
    assert corr.shape[0] == corr.shape[1], "corr должна быть квадратной матрицей"
    assert corr.shape[0] == len(lambdas), "Размер corr должен соответствовать количеству lambdas"
    
    # Проверка симметричности
    assert np.allclose(corr, corr.T), "Корреляционная матрица должна быть симметричной"
    
    # Проверка диагонали
    assert np.allclose(np.diag(corr), 1), "Диагональные элементы corr должны быть равны 1"
    
    # Проверка допустимых значений корреляции
    assert np.all(corr >= -1) and np.all(corr <= 1), "Корреляции должны быть в диапазоне [-1, 1]"
    
    # Проверка положительной полуопределённости (упрощённая)
    try:
        np.linalg.cholesky(corr + 1e-10*np.eye(len(corr)))
    except np.linalg.LinAlgError:
        assert False, "Корреляционная матрица должна быть положительно полуопределённой"
    
    print("Success")

def debug_plot(Z, Y, U, X, factor, debug=False):
    if debug:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = iter(axes.flatten())

        ax = next(axes)
        sns.scatterplot(x=Z[0], y=Z[1], ax=ax), ax.set_title("Z[0], Z[1] distribution")
        ax = next(axes)
        sns.scatterplot(x=Y[0], y=Y[1], ax=ax), ax.set_title("Y[0], Y[1] distribution")
        ax = next(axes)
        sns.scatterplot(x=U[0], y=U[1], ax=ax), ax.set_title("U[0], U[1] distribution")
        ax = next(axes)
        sns.histplot(x=(X.sum(axis=0) > 1+factor), ax=ax), ax.set_title("X distribution")
        plt.show()
    return 

# COMMAND ----------

from scipy.stats import norm, poisson
# fix correlation diag from zeros to ones
def fix_diag(corr):
    if np.sum(corr * np.eye(len(corr))) == 0:
        corr = corr + np.eye(len(corr))
        print("diag fixed")
    return corr

# Poisson reach simulation
def simulate_reach(lambdas, corr, T_active=None, T=4*3600, p_listen=0.2,  conservative_factor=0, n_draws=10_000, seed=0, debug=False):
    k = len(lambdas)
    lambdas = lambdas / p_listen
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    # 1. covariance & Cholesky
    corr = fix_diag(corr)
    do_assert(lambdas, corr)

    cov = np.outer(np.sqrt(lambdas), np.sqrt(lambdas)) * corr
    L = np.linalg.cholesky(cov + 1e-6*np.eye(k))   # jitter for PD
    # 2. Gaussian copula → Poisson
    Z = rng.standard_normal((k, n_draws))
    Y = L @ Z
    #    2.1 Normalization
    std_Y = np.std(Y, axis=1, keepdims=True)
    Y = Y / std_Y

    U = norm.cdf(Y)
    X = poisson.ppf(U, lambdas[:,None]).astype(int)

    # 3. Zero probability - some people don't listen to any channel
    # is_zero = np.random.rand(k, n_draws) > p_listen
    # X[is_zero] = 0

    # 4. Combined reach
    reach = (X.sum(axis=0) > 1 + conservative_factor * k).mean()
    debug_plot(Z=Z, Y=Y, U=U, X=X, factor=conservative_factor * k, debug=debug)
    return reach * p_listen

simulate_reach(lambdas, corr, T, debug=False) #* POPULATION['AkronOH']

# COMMAND ----------

# MAGIC %md
# MAGIC #### Gurobi schedule optimisation

# COMMAND ----------

model = gp.Model("MediaOptimization")

model.setParam('MIPGap', 0.1)      # Приемлемая погрешность 10%
model.setParam('TimeLimit', 60)     # Максимум 1 минута
model.setParam('Heuristics', 0.05) # Минимум эвристик
model.setParam('Cuts', 0)           # Без отсечений

# Main variables
x = model.addVars(len(lambdas), vtype=gp.GRB.INTEGER, lb=0, ub=min(max_posts, 10), name="Posts")
reach = model.addVars(len(lambdas), name="Reach") 

# Audience intersection constrains min(reach[i], reach[j])
min_intersect = {}
for i in range(len(lambdas)):
    for j in range(i+1, len(lambdas)):
        min_ij = model.addVar(name=f"min_{i}_{j}")
        model.addGenConstrMin(min_ij, [reach[i], reach[j]], name=f"min_constr_{i}_{j}")
        min_intersect[(i,j)] = min_ij

# Target function taking into account unique coverage
unique_coef = 0.3
# Total reach 
total_reach = (sum(reach[j] for j in range(len(lambdas))) 
               - unique_coef * sum(min_intersect[(i,j)] * C[i,j] 
                                   for i in range(len(lambdas)) 
                                   for j in range(i+1, len(lambdas))
                                   )
               )

# Goal function
model.setObjective(total_reach, gp.GRB.MAXIMIZE)

N_dots = max_posts + 1
conservative_factor = 0.9

# Non-linear increment for reach: reach[j] = cume[j] * √x[j]
for j in range(len(lambdas)):
    x_points = list(range(N_dots + 1))
    y_points = [lambdas[j] * conservative_factor * (n)**(1/10) for n in x_points] 
    model.addGenConstrPWL(x[j], reach[j], x_points, y_points, f"PWL_Sqrt_{j}")

# No more than 150% of origin reach
for j in range(len(lambdas)):
    model.addConstr(reach[j] <= lambdas[j]* 1.5)

# no more than max_posts publication by channel-dow-daypart
for j in range(len(lambdas)):
    model.addConstr(x[j] <= max_posts)  

# Budget constrains
model.addConstr(sum(cost[j] * x[j] for j in range(len(lambdas))) <= budget)

# No more than N publication 
model.addConstr(sum(x[j] for j in range(len(lambdas))) <= len(lambdas) * 3)
model.optimize()

result = np.empty((len(lambdas),2))
# Show results
print("Amount of puplication optimisation:")
for j in range(len(lambdas)):
    print(f"Channel-dow-daypart schedule slot {j+1}: {int(x[j].X)} publications, reach = {reach[j].X:.3f}")
    result[j, 0] = x[j].X
    result[j, 1] = reach[j].X
print(f"Gurobi reach: {model.ObjVal:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Output (for budget = 20 000)

# COMMAND ----------

# show schedule optimisation result
df_plot = (schedule.assign(posts=result[:,0], reach_model=result[:,1])
#  .assign(budget = lambda x: x['population'] * x['reach'] / 1000 * x['cpm'])['budget'].sum()
# .assign(budget = lambda x: x['aqh_cost'] * x['posts'])#['budget'].sum()
# ['reach'].sum()
# .sort_values(['posts', 'reach'], ascending=False)

)
print('reach_without correlation (Gurobi model):', df_plot['reach_model'].sum(), "spots count:", df_plot['posts'].sum())
df_plot#.loc[lambda x: x['reach_model'] > 1e-5]

# COMMAND ----------

# estimate combined reach based on optimial schedule
reach_model = df_plot['reach_model'].to_numpy().flatten()
reach = simulate_reach(reach_model, corr, T, debug=False) 
print("Reach share (Poisson reach estimation)", reach, "Absolute reach (Poisson reach estimation)", reach * POPULATION['AkronOH'])

# COMMAND ----------

# non-linear reach increment visualisation
'''
    #[2 / (1 + np.exp(-lambdas[j] * n)) - 1 for n in x_points] 
    #[lambdas[j] * (n)**(1/3) for n in x_points]  
    #[(lambdas[j] * n) / (1 + lambdas[j] * n) for n in x_points]
    #[np.tanh(lambdas[j] * n) for n in x_points] 
    #[lambdas[j] * np.sqrt(n) for n in x_points]
'''
x_points = [0,0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1]
y_points = [2 / (1 + np.exp(- n)) - 1 for n in x_points]
sns.lineplot(x=x_points, y=y_points, color = 'blue', label = 'logit')
y_points = [ (n)**(1/3) for n in x_points]  
sns.lineplot(x=x_points, y=y_points, color = 'green', label = 'x^1/3')
y_points = [ np.sqrt(n) for n in x_points]
sns.lineplot(x=x_points, y=y_points, color = 'red', label = 'x^1/2')
y_points = [np.tanh( n) for n in x_points] 
sns.lineplot(x=x_points, y=y_points, color = 'black', label = 'tan')
y_points = [( n) / (1 + n) for n in x_points]
sns.lineplot(x=x_points, y=y_points, color = 'orange', label = 'x/(1+x)')
y_points = [ (n)**(1/10) for n in x_points]  
sns.lineplot(x=x_points, y=y_points, color = 'yellow', label = 'x^1/5')
plt.show()

# COMMAND ----------

# show differrence between non-linear function 
n = 3
j = 0 
print("x, n*x \t\t\t", lambdas[0], lambdas[0] * n)
print("1/1+x \t\t\t",  lambdas[j] * n /(1+lambdas[j] * n),  )
print("x^1/10 \t\t\t", lambdas[j] *  (n)**(1/10))
print("2 / (1 + exp(-x) - 1) \t", 2 / (1 + np.exp(-lambdas[0] * n)) - 1  )
print("x^1/10 \t\t\t", lambdas[j] * np.sqrt(n) )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optimisation (Full Pipeline)

# COMMAND ----------

# define Gurobi optimisation  function
def schedule_optimisation(lambdas, corr, budget, max_posts):
    model = gp.Model("MediaOptimization")

    model.setParam('MIPGap', 0.1)      # Приемлемая погрешность 10%
    model.setParam('TimeLimit', 60)     # Максимум 1 минута
    model.setParam('Heuristics', 0.05) # Минимум эвристик
    model.setParam('Cuts', 0)           # Без отсечений

    model = gp.Model("MediaOptimization")
    # Переменные
    x = model.addVars(len(lambdas), vtype=gp.GRB.INTEGER, lb=0, ub=min(max_posts, 10), name="Posts")
    reach = model.addVars(len(lambdas), name="Reach") 

    # Пересечение аудиторий через min(reach[i], reach[j])
    min_intersect = {}
    for i in range(len(lambdas)):
        for j in range(i+1, len(lambdas)):
            min_ij = model.addVar(name=f"min_{i}_{j}")
            model.addGenConstrMin(min_ij, [reach[i], reach[j]], name=f"min_constr_{i}_{j}")
            min_intersect[(i,j)] = min_ij

    # Целевая функция с учётом уникального охвата
    unique_coef = 0.3
    # Total reach 
    total_reach = (sum(reach[j] for j in range(len(lambdas))) 
                - unique_coef * sum(min_intersect[(i,j)] * C[i,j] 
                                    for i in range(len(lambdas)) 
                                    for j in range(i+1, len(lambdas))
                                    )
                )

    # Целевая функция: максимизировать охват
    model.setObjective(total_reach, gp.GRB.MAXIMIZE)

    N_dots = max_posts + 1
    conservative_factor = 0.9
    # Нелинейный прирост охвата: reach[j] = cume[j] * √x[j]
    for j in range(len(lambdas)):
        x_points = list(range(N_dots + 1))
        y_points = [lambdas[j] * conservative_factor * (n)**(1/10) for n in x_points] 
        model.addGenConstrPWL(x[j], reach[j], x_points, y_points, f"PWL_Sqrt_{j}")

    # Ни один канал не даст больше 150% от своего охвата
    for j in range(len(lambdas)):
        model.addConstr(reach[j] <= lambdas[j]* 1.5)#max(lambdas.flatten() * 1.5)) 

    # Не больше max_posts публикаций на канал
    for j in range(len(lambdas)):
        model.addConstr(x[j] <= max_posts)  

    # Ограничение на бюджет
    model.addConstr(sum(cost[j] * x[j] for j in range(len(lambdas))) <= budget)

    # не более N публикаций на все каналы
    model.addConstr(sum(x[j] for j in range(len(lambdas))) <= len(lambdas) * 3)

    model.optimize()

    result = np.empty((len(lambdas),2))
    # Вывод результатов
    print("Оптимальное количество публикаций:")
    for j in range(len(lambdas)):
        #print(f"Канал {j+1}: {int(x[j].X)} публикаций, охват = {reach[j].X:.3f}")
        result[j, 0] = x[j].X
        result[j, 1] = reach[j].X
    print(f"Общий охват: {model.ObjVal:.3f}")
    return result

####################################################### FULL Pipeline #######################################################
budget_increment = 5_000
budget_data = []
reach_data = []
schedule_data = []

# find optimal schedule for different budgets
for i in range(1, 20):
    # define budget
    budget =  budget_increment*i
    # find optimal schedule for current budget
    optimal_schedule = schedule_optimisation(lambdas=lambdas, corr=C, budget=budget, max_posts=3)
    # get lambdas from current budget
    reach_model = optimal_schedule[:, 1]
    # estimate combined reach based on optimial schedule 
    combined_reach = simulate_reach(reach_model, corr, debug=False) 
    # save data
    budget_data.append(budget)
    reach_data.append(combined_reach)
    schedule_data.append(optimal_schedule)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Output

# COMMAND ----------

# plot budget and reach relation
ax = sns.lineplot(x=budget_data[:-2], y=np.array(reach_data[:-2]))
ax.axvline(x=budget_data[6], ymin=0, ymax=1)
ax.set_title("Budget and Reach relation")
ax.set_xlabel("Budget")
ax.set_ylabel("Reach")

# COMMAND ----------

# show optimal schedule for budget = 35_000
(schedule.assign(posts=schedule_data[6][:,0], reach_model=schedule_data[6][:,1])
.loc[lambda x: x['posts'] > 1e-5])

# COMMAND ----------

# show combined reach
combined_reach[6]

# COMMAND ----------

# MAGIC %md
# MAGIC # OLD

# COMMAND ----------

# MAGIC %md
# MAGIC ### old

# COMMAND ----------

model = gp.Model("MediaOptimization")

model.setParam('MIPGap', 0.1)      # Приемлемая погрешность 10%
model.setParam('TimeLimit', 60)     # Максимум 1 минута
model.setParam('Heuristics', 0.05) # Минимум эвристик
model.setParam('Cuts', 0)           # Без отсечений

# Переменные
x = model.addVars(len(lambdas), vtype=gp.GRB.INTEGER, lb=0, ub=min(max_posts, 10), name="Posts")
reach = model.addVars(len(lambdas), name="Reach") 

# Целевая функция с учётом уникального охвата
unique_coef = 5
total_reach = (sum(reach[j] for j in range(len(lambdas))) - 
              unique_coef * sum(reach[i]*reach[j]*C[i,j] 
                                for i in range(len(lambdas)) 
                                for j in range(i+1, len(lambdas))))

# Целевая функция: максимизировать охват
model.setObjective(total_reach, gp.GRB.MAXIMIZE)

N_dots = 10
conservative_factor = 0.8
# Нелинейный охват: reach[j] = cume[j] * √x[j]
for j in range(len(lambdas)):
    x_points = list(range(N_dots + 1))
    y_points = [conservative_factor*(lambdas[j] * n) / (1 + lambdas[j] * n) for n in x_points] #[lambdas[j] * conservative_factor * (n)**(1/5) for n in x_points] 
    model.addGenConstrPWL(x[j], reach[j], x_points, y_points, f"PWL_Sqrt_{j}")

# Ни один канал не даст больше 150% от своего охвата
for j in range(len(lambdas)):
    model.addConstr(reach[j] <= lambdas[j]* 1.5)#max(lambdas.flatten() * 1.5)) 

# Не больше max_posts публикаций на канал
for j in range(len(lambdas)):
    model.addConstr(x[j] <= max_posts)  

# Ограничение на бюджет
model.addConstr(sum(cost[j] * x[j] for j in range(len(lambdas))) <= budget)

# не более N публикаций на все каналы
model.addConstr(sum(x[j] for j in range(len(lambdas))) <= len(lambdas) )

model.optimize()

result = np.empty((len(lambdas),2))
# Вывод результатов
print("Оптимальное количество публикаций:")
for j in range(len(lambdas)):
    print(f"Канал {j+1}: {int(x[j].X)} публикаций, охват = {reach[j].X:.3f}")
    result[j, 0] = x[j].X
    result[j, 1] = reach[j].X
print(f"Общий охват: {model.ObjVal:.3f}")

# COMMAND ----------

model = gp.Model("MediaOptimization")

# Переменные
x = model.addVars(len(lambdas), vtype=gp.GRB.INTEGER, lb=min_posts, ub=max_posts, name="Posts")
reach = model.addVars(len(lambdas), name="Reach")  # P_j = 1 - exp(-λ_j * x_j)

N_dots = 4
# Кусочно-линейная аппроксимация для P_j
for j in range(len(lambdas)):
    x_points = list(range(N_dots + 1))
    y_points = [1 - np.exp(-lambdas[j] * n) for n in x_points]
    model.addGenConstrPWL(x[j], reach[j], x_points, y_points)

# Линейная часть целевой функции: сумма P_j
linear_part = sum(reach[j] for j in range(len(lambdas)))

# Квадратичная поправка: сумма P_i * P_j * ρ_ij
quad_correction = 0
for i in range(len(lambdas)):
    for j in range(i + 1, len(lambdas)):
        quad_correction += reach[i] * reach[j] * C[i, j]

# Общий охват: linear_part - quad_correction
total_reach = linear_part - quad_correction

# Целевая функция: максимизировать охват
model.setObjective(total_reach, gp.GRB.MAXIMIZE)

# Ограничение на бюджет
model.addConstr(sum(cost[j] * x[j] for j in range(len(lambdas))) <= budget)

model.optimize()

result = np.empty((len(lambdas),2))
# Вывод результатов
print("Оптимальное количество публикаций:")
for j in range(len(lambdas)):
    print(f"Канал {j+1}: {int(x[j].X)} публикаций, охват = {reach[j].X:.3f}")
    result[j, 0] = x[j].X
    result[j, 1] = reach[j].X
print(f"Общий охват: {model.ObjVal:.3f}")

# COMMAND ----------

(schedule.assign(posts=result[:,0], reach_=result[:,1])
#  .assign(budget = lambda x: x['population'] * x['reach'] / 1000 * x['cpm'])['budget'].sum()
# ['reach'].sum()
.sort_values(['posts', 'reach'], ascending=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Attempt 1

# COMMAND ----------

print('correlation matrix shape: ', corr.shape, 'reach shape:', lambdas.shape)
import numpy as np
from scipy.stats import norm, poisson
from scipy.linalg import ldl

def modified_cholesky(A):
    L, D, perm = ldl(A, hermitian=True)
    D = np.diag(D)
    D = np.maximum(D, 1e-12)
    return L @ np.diag(np.sqrt(D))

def nearest_pd(A):
    """Находит ближайшую положительно определённую матрицу"""
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = V.T @ np.diag(s) @ V
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    return A3


def simulate_reach(lambdas, corr, n_draws=200_000, seed=0):
    k = len(lambdas)
    rng = np.random.default_rng(seed)
    # 1. covariance & Cholesky
    cov = np.outer(np.sqrt(lambdas), np.sqrt(lambdas)) * corr
    cov = nearest_pd(cov)
    L = np.linalg.cholesky(cov + 1e-12*np.eye(k))  # jitter for PD
    # 2. Gaussian copula → Poisson
    Z = rng.standard_normal((k, n_draws))
    Y = L @ Z
    U = norm.cdf(Y)
    X = poisson.ppf(U, lambdas[:,None]).astype(int)
    # 3. Combined reach
    reach = (X.sum(axis=0) > 0).mean()
    return reach

# example
print(simulate_reach(lambdas, corr))

# COMMAND ----------

import numpy as np
from scipy.stats import norm, poisson

def simulate_reach(lambdas, corr, n_draws=200_000, seed=0):
    k = len(lambdas)
    rng = np.random.default_rng(seed)
    # 1. covariance & Cholesky
    cov = np.outer(np.sqrt(lambdas), np.sqrt(lambdas)) * corr
    L = np.linalg.cholesky(cov + 1e-12*np.eye(k))   # jitter for PD
    # 2. Gaussian copula → Poisson
    Z = rng.standard_normal((k, n_draws))
    Y = L @ Z
    U = norm.cdf(Y)
    X = poisson.ppf(U, lambdas[:,None]).astype(int)
    # 3. Combined reach
    reach = (X.sum(axis=0) > 0).mean()
    return reach

# example
lambdas = np.array([0.15, 0.10, 0.08])   # channel means
corr     = np.array([[1, .50, .55],
                     [.50, 1, .55],
                     [.55, .55, 1]])
print(simulate_reach(lambdas, corr))

# COMMAND ----------

sum(lambdas * 1_000_000), 5.5e-05 * 1_000_000

# COMMAND ----------

sns.histplot(corr.flatten())
plt.show()
sns.histplot(lambdas.flatten())

# COMMAND ----------

eps = 0.001
corr_matrix = (data.assign(cpm = lambda x: x['aqh_cost'] / x['aqh'])
     .merge(data.drop(columns=['aqh_cost']), on=['market'], how='left')
     .loc[lambda x: ~((x['channel_x'] == x['channel_y']) & (x['dow_x'] == x['dow_y']) & (x['daypart_x'] == x['daypart_y']))]
     .assign(aqh_min = lambda x: np.minimum(x['aqh_x'], x['aqh_y']))
     .assign(rnd = lambda x: np.random.uniform(0.01, 1, len(x)))
     .assign(aqh_pairvise = lambda x: (x['rnd'] * x['aqh_min']).astype(int))
     .assign(duplication_prc = lambda x: x['aqh_pairvise'] / x['aqh_min'] )
     .assign(adh_combined = lambda x: x['aqh_x'] + x['aqh_y'])
     .assign(combined_cume = lambda x: x['cume_x'] + x['cume_y'])
     .assign(corr = lambda x: x['duplication_prc'] )
     .assign(x = lambda x: x[['channel_x',  'dow_x', 'daypart_x']].astype(str).agg('_'.join, axis=1) )
     .assign(y = lambda x: x[['channel_y',  'dow_y', 'daypart_y']].astype(str).agg('_'.join, axis=1) )
     [['x', 'y', 'corr']]
     .pivot_table(index='x', columns='y', values='corr', fill_value=1, aggfunc='sum')
     .apply(lambda x: np.where(x==0, eps, x))
)
corr_matrix

# COMMAND ----------

lambda_data = (data.assign(population = lambda x: x['market'].map(POPULATION))#
     .assign(lmd = lambda x: x['cume'] / x['population'])
     .assign(x = lambda x: x[['channel',  'dow', 'daypart']].astype(str).agg('_'.join, axis=1) )
     .set_index('x')
     .reindex(corr_matrix.index)
     [['lmd']]
 )
lambda_data

# COMMAND ----------



# COMMAND ----------

import numpy as np
from scipy.stats import norm, poisson

def simulate_reach(lambdas, corr, n_draws=200_000, seed=0):
    k = len(lambdas)
    rng = np.random.default_rng(seed)
    # 1. covariance & Cholesky
    cov = np.outer(np.sqrt(lambdas), np.sqrt(lambdas)) * corr
    L = np.linalg.cholesky(cov + 1e-12*np.eye(k))   # jitter for PD
    # 2. Gaussian copula → Poisson
    Z = rng.standard_normal((k, n_draws))
    Y = L @ Z
    U = norm.cdf(Y)
    X = poisson.ppf(U, lambdas[:,None]).astype(int)
    # 3. Combined reach
    reach = (X.sum(axis=0) > 0).mean()
    return reach

# example
lambdas = lambda_data.to_numpy()
corr    = corr_matrix.to_numpy()
print(simulate_reach(lambdas, corr))