# Projeto Bank Churn Prediction

Este projeto tem como objetivo realizar uma análise exploratória dos dados e aplicar modelos de Machine Learning para prever resultados. Abaixo estão os passos detalhados do processo.

## Etapas do Projeto

### 1. Download dos Datasets e Importação das Bibliotecas
- Baixei os datasets necessários.
- Importe as bibliotecas essenciais para a análise de dados e Machine Learning, como `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, entre outras.

### 2. Análise Geral dos Dados
- Carreguei os datasets no ambiente de desenvolvimento.
- Realizei uma análise geral dos dados para entender suas características principais.
  - Verifiquei a estrutura dos dados, tipos de variáveis, e valores ausentes.
  - Visualizei algumas estatísticas descritivas básicas.

### 3. Separação dos Dados de Treino e Teste
- Separei os dados em conjuntos de treino e teste utilizando a função `train_test_split` do `scikit-learn`.

### 4. Pré-processamento dos Dados
- Apliquei métodos estatísticos e de Análise Exploratória de Dados (EDA) para entender melhor as distribuições e relações entre as variáveis.
- Transformei variáveis categóricas em variáveis numéricas utilizando encoders, como o `OneHotEncoder`.

### 5. Feature Engineering
- Criei novas colunas a partir das variáveis existentes para potencialmente melhorar o desempenho dos modelos.
- Apliquei as mesmas modificações no conjunto de dados de teste para garantir a consistência.

### 6. Modelagem de Machine Learning
- Apliquei os seguintes modelos de Machine Learning padrões:
  - Regressão Logística
  - Random Forest
  - Multi-Layer Perceptron (MLP)

### 7. Otimização de Hiperparâmetros
- Realizei a busca de hiperparâmetros para os modelos utilizando técnicas como `GridSearchCV` ou `RandomizedSearchCV`.
- Foquei na otimização da métrica ROC AUC.

### 8. Avaliação e Visualização dos Resultados
- Avaliei os modelos com base na métrica ROC AUC.
- Plotei gráficos para visualização dos resultados, como:
   ![Curva Roc MLP](https://github.com/guievbs/bank-churn-prediction/blob/main/report/images/roc_auc_mlp.png)
   ![Curva Roc RF](https://github.com/guievbs/bank-churn-prediction/blob/main/report/images/roc_auc_rf.png)

## Conclusão
- Os resultados não foram satisfatórios, provavelmente devido ao overfitting dos dados de treino.
- O desempenho dos modelos nos dados de teste foi baixo, indicando que os modelos não generalizaram bem para novos dados.

## Como Reproduzir Este Projeto
1. Clone este repositório.
2. Instale as dependências listadas em `requirements.txt`.
3. Execute o notebook ou script principal para reproduzir a análise e os modelos.

