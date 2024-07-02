import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
sns.set_palette("pastel")

def plot_roc_curve(model, X_train, X_test, y_train, y_test, model_name) -> float:
    """
    Plota a curva ROC para um modelo dado usando Seaborn para estilização.

    Parâmetros:
    - model: modelo treinado do scikit-learn.
    - X_train: dados de treino.
    - X_test: dados de teste.
    - y_train: rótulos de treino.
    - y_test: rótulos de teste.
    - model_name: nome do modelo para exibição no título do gráfico.
    """
    # Treina o modelo
    model.fit(X_train, y_train)
    
    # Obtem as probabilidades previstas
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcula as taxas de falso positivo e verdadeiro positivo
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    

    # Ajustar o layout
    plt.figure(figsize=(8, 6))
    plt.tight_layout()
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos', labelpad=15)
    plt.ylabel('Taxa de Verdadeiros Positivos', labelpad=15)
    plt.title(f'{model_name}', pad=12)
    plt.legend(loc="lower right")
    sns.despine(offset=10, trim=True) 
    plt.show()

    return roc_auc
