# %%
# Importação de bibliotecas
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

#%%
# Carregar Dados
df = pd.read_csv("../dataset/insurance.csv")
df.head()
# %%
# Pré-Processamento
categorical_col = ['sex', 'smoker', 'region']
numerical_col = ['age', 'bmi', 'children']
target = 'charges'

X = df[numerical_col+categorical_col]
y = df[target]

preprocessor = ColumnTransformer(
    transformers= [
        # (Nome do processamento, Função a ser aplicada, colunas que sofrerão alteração)
        ('num', StandardScaler(), numerical_col),
        ('cat', OneHotEncoder(), categorical_col)
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

#%%
# Validação Holdout
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
# %%
# Predição do modelo
y_pred = pipeline.predict(X_test)
# %%
# Validar os dados preditos
MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)

print(f"MAE: {MAE:.2f}")
print(f"MSE: {MSE:.2f}")
print(f"R2: {R2:.2f}")
# %%
# Validação Holdout com 3 splits
# separar 20% para teste
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
# Separar outros 20% para validação (25%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
pipeline.fit(X_train, y_train)
#%%
y_pred = pipeline.predict(X_val)
#%%
MAE = mean_absolute_error(y_val, y_pred)
MSE = mean_squared_error(y_val, y_pred)
R2 = r2_score(y_val, y_pred)

print(f"MAE: {MAE:.2f}")
print(f"MSE: {MSE:.2f}")
print(f"R2: {R2:.2f}")
#%%
y_pred = pipeline.predict(X_test)
MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)

print(f"MAE: {MAE:.2f}")
print(f"MSE: {MSE:.2f}")
print(f"R2: {R2:.2f}")
#%%
# Validação cruzada
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')

print(f"cv_scores: {cv_scores}")
print(f"Média cv_scores: {np.mean(cv_scores)}")