# Validação de Modelos de Machine Learning

Este repositório contém um projeto de aprendizado sobre validação de modelos de Machine Learning. O objetivo principal é demonstrar o uso de diferentes técnicas de validação, como **validação Holdout** e **validação cruzada (K-Fold)**, utilizando **Regressão Linear** no conjunto de dados `insurance.csv`.

---

## 📌 Descrição

O projeto utiliza um conjunto de dados (`insurance.csv`) com informações sobre clientes e os valores cobrados pelos seguros. O objetivo é prever a variável `charges` (cobranças) com base em atributos como idade (`age`), índice de massa corporal (`bmi`), número de filhos (`children`), sexo (`sex`), se é fumante (`smoker`) e região (`region`).

### 📊 **1. Carregamento e Pré-processamento de Dados**  
Os dados são carregados e separados em **variáveis categóricas** e **numéricas**. Para garantir que os dados fiquem no formato adequado para o modelo, aplicamos **One-Hot Encoding** às colunas categóricas e **padronização (StandardScaler)** às numéricas:

```python
df = pd.read_csv("../dataset/insurance.csv")

categorical_col = ['sex', 'smoker', 'region']
numerical_col = ['age', 'bmi', 'children']
target = 'charges'

X = df[numerical_col + categorical_col]
y = df[target]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_col),
        ('cat', OneHotEncoder(), categorical_col)
    ]
)
```

### 🤖 **2. Construção do Pipeline e Treinamento do Modelo**  
Criamos um **Pipeline** para aplicar o pré-processamento e treinar um modelo de **Regressão Linear**:

```python
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)
```

---

## 🔎 **Técnicas de Validação Utilizadas**

### 1º **Validação Holdout**  
Dividimos os dados em **80% treino** e **20% teste** para avaliar o modelo:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)

print(f"MAE: {MAE:.2f}")
print(f"MSE: {MSE:.2f}")
print(f"R2: {R2:.2f}")
```

### 2º **Validação Holdout com 3 Splits**  
Avaliamos o modelo separando **20% para teste** e dentro do conjunto de treino, reservamos mais **25% para validação**:

```python
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_val)
print(f"MAE Validação: {mean_absolute_error(y_val, y_pred):.2f}")
print(f"MSE Validação: {mean_squared_error(y_val, y_pred):.2f}")
print(f"R2 Validação: {r2_score(y_val, y_pred):.2f}")
```

### 3º **Validação Cruzada (K-Fold)**  
Para garantir a robustez do modelo, utilizamos **validação cruzada com 5 folds**:

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')

print(f"cv_scores: {cv_scores}")
print(f"Média cv_scores: {np.mean(cv_scores)}")
```

---

## 📈 **Resultados**

Após as avaliações, observamos que a validação cruzada ajuda a reduzir o risco de overfitting e fornece uma métrica mais confiável do desempenho do modelo.

### 📋 **Métricas Utilizadas**
- **MAE (Mean Absolute Error)**: Média dos erros absolutos.
- **MSE (Mean Squared Error)**: Erro médio quadrático.
- **R² Score**: Indica o quão bem o modelo explica a variabilidade dos dados.

---

## 🚀 **Como Executar o Projeto**

1⃣ Clone este repositório:
```bash
git clone https://github.com/username/repository-name.git
```

2⃣ Instale as dependências:
```bash
pip install -r requirements.txt
```

3⃣ Execute o código em um Jupyter Notebook ou diretamente em um script Python.

---

## 💡 **Conclusão**
Este projeto ajudou a compreender como avaliar modelos de Machine Learning usando diferentes técnicas de validação. A aplicação dessas estratégias é essencial para garantir que um modelo seja generalizável e confiável ao lidar com novos dados.
