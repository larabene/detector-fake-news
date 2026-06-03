import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Carrega os modelos
with open("modelos.pkl", "rb") as f:
    dados = pickle.load(f)

modelos_treinados = dados["modelos"]
X_test = dados["X_test"]
y_test = dados["y_test"]

# Cor diferente para cada modelo
cores = {
    'Naive Bayes':    'Purples',
    'Reg. Logística': 'Blues',
    'SVM':            'Greens',
    'Random Forest':  'Oranges'
}

f1_scores = {
    'Naive Bayes':    87.11,
    'Reg. Logística': 93.77,
    'SVM':            94.59,
    'Random Forest':  93.49
}

modelos_ordem = ['Naive Bayes', 'Reg. Logística', 'SVM', 'Random Forest']
posicoes = [(0,0), (0,1), (1,0), (1,1)]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Matriz de Confusão — Comparação entre os 4 Modelos',
             fontsize=14, fontweight='bold', y=1.02)

for nome, pos in zip(modelos_ordem, posicoes):
    modelo = modelos_treinados[nome][0]
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    ax = axes[pos[0]][pos[1]]

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=cores[nome],
        xticklabels=['Fake', 'Verdadeira'],
        yticklabels=['Fake', 'Verdadeira'],
        ax=ax,
        annot_kws={"size": 13}
    )

    destaque = ' ★ Melhor modelo' if nome == 'SVM' else ''
    ax.set_title(f'{nome}{destaque}\nF1-Score: {f1_scores[nome]}%',
                fontsize=11, fontweight='bold')
    ax.set_ylabel('Valor Real', fontsize=10)
    ax.set_xlabel('Valor Previsto', fontsize=10)

    erros = cm[0][1] + cm[1][0]
    acertos = cm[0][0] + cm[1][1]
    ax.text(0.5, -0.18,
            f'Acertos: {acertos} | Erros: {erros} de {len(y_test)} notícias',
            transform=ax.transAxes,
            ha='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig('matrizes_todos_modelos.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Gráfico salvo como 'matrizes_todos_modelos.png'!")