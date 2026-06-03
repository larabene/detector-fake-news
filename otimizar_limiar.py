# ================================================
# otimizar_limiar.py — versão corrigida
# ================================================

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (f1_score, precision_score,
                             recall_score, accuracy_score)
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ================================================
# Carrega o modelo E o conjunto de teste original
# ================================================
print("Carregando modelos e conjunto de teste...")
with open("modelos.pkl", "rb") as f:
    dados_salvos = pickle.load(f)

modelos_treinados = dados_salvos["modelos"]
vectorizer = dados_salvos["vectorizer"]
X_test = dados_salvos["X_test"]
y_test = dados_salvos["y_test"]

print(f"✓ Conjunto de teste carregado: {X_test.shape[0]} notícias")
print(f"  Fake: {sum(y_test==0)} | Verdadeiras: {sum(y_test==1)}")

pesos = {
    'Naive Bayes':    1,
    'Reg. Logística': 2,
    'SVM':            3,
    'Random Forest':  2
}

limiares = np.arange(0.50, 0.96, 0.01)

# ================================================
# Testa limiares por modelo individual
# ================================================
print("\n" + "="*60)
print("LIMIAR IDEAL POR MODELO (maximizando F1-Score)")
print("="*60)

melhores_limiares = {}

for nome, (modelo, _) in modelos_treinados.items():
    probs = modelo.predict_proba(X_test)[:, 1]

    melhor_f1 = 0
    melhor_limiar = 0.5
    melhor_acc = melhor_prec = melhor_rec = 0

    for limiar in limiares:
        preds = (probs >= limiar).astype(int)
        if len(np.unique(preds)) < 2:
            continue
        f1 = f1_score(y_test, preds, zero_division=0)
        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_limiar = limiar
            melhor_acc  = accuracy_score(y_test, preds) * 100
            melhor_prec = precision_score(y_test, preds, zero_division=0) * 100
            melhor_rec  = recall_score(y_test, preds, zero_division=0) * 100

    melhores_limiares[nome] = melhor_limiar
    print(f"\n{nome}:")
    print(f"  Limiar ideal:  {melhor_limiar:.2f} ({melhor_limiar*100:.0f}%)")
    print(f"  Acurácia:      {melhor_acc:.2f}%")
    print(f"  Precisão:      {melhor_prec:.2f}%")
    print(f"  Recall:        {melhor_rec:.2f}%")
    print(f"  F1-Score:      {melhor_f1*100:.2f}%")

# ================================================
# Testa limiares para o sistema de voto ponderado
# ================================================
print("\n" + "="*60)
print("LIMIAR IDEAL PARA O SISTEMA DE VOTO PONDERADO")
print("="*60)

melhor_f1_sistema = 0
melhor_limiar_sistema = 0.5
melhor_acc_sistema = melhor_prec_sistema = melhor_rec_sistema = 0
resultados_limiares = []

for limiar in limiares:
    votos_verdadeira = np.zeros(X_test.shape[0])
    votos_fake = np.zeros(X_test.shape[0])

    for nome, (modelo, _) in modelos_treinados.items():
        probs = modelo.predict_proba(X_test)[:, 1]
        peso = pesos[nome]
        for i, prob in enumerate(probs):
            if prob >= limiar:
                votos_verdadeira[i] += peso
            else:
                votos_fake[i] += peso

    preds_sistema = (votos_verdadeira > votos_fake).astype(int)

    if len(np.unique(preds_sistema)) < 2:
        continue

    f1   = f1_score(y_test, preds_sistema, zero_division=0)
    acc  = accuracy_score(y_test, preds_sistema) * 100
    prec = precision_score(y_test, preds_sistema, zero_division=0) * 100
    rec  = recall_score(y_test, preds_sistema, zero_division=0) * 100

    resultados_limiares.append({
        'limiar': limiar, 'f1': f1*100,
        'acuracia': acc, 'precisao': prec, 'recall': rec
    })

    if f1 > melhor_f1_sistema:
        melhor_f1_sistema = f1
        melhor_limiar_sistema = limiar
        melhor_acc_sistema  = acc
        melhor_prec_sistema = prec
        melhor_rec_sistema  = rec

print(f"\nLimiar ideal: {melhor_limiar_sistema:.2f} ({melhor_limiar_sistema*100:.0f}%)")
print(f"  Acurácia:  {melhor_acc_sistema:.2f}%")
print(f"  Precisão:  {melhor_prec_sistema:.2f}%")
print(f"  Recall:    {melhor_rec_sistema:.2f}%")
print(f"  F1-Score:  {melhor_f1_sistema*100:.2f}%")

# ================================================
# Gráfico
# ================================================
df_limiares = pd.DataFrame(resultados_limiares)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(df_limiares['limiar']*100, df_limiares['f1'],
         color='#7F77DD', linewidth=2, label='F1-Score')
plt.plot(df_limiares['limiar']*100, df_limiares['acuracia'],
         color='#1D9E75', linewidth=2, label='Acurácia')
plt.plot(df_limiares['limiar']*100, df_limiares['precisao'],
         color='#D85A30', linewidth=2, label='Precisão')
plt.plot(df_limiares['limiar']*100, df_limiares['recall'],
         color='#BA7517', linewidth=2, label='Recall')
plt.axvline(x=melhor_limiar_sistema*100, color='red',
            linestyle='--', linewidth=1.5,
            label=f'Limiar ideal ({melhor_limiar_sistema*100:.0f}%)')
plt.axvline(x=65, color='gray', linestyle=':',
            linewidth=1, label='Limiar atual (65%)')
plt.xlabel('Limiar de confiança (%)')
plt.ylabel('Valor de Acerto(%)')
plt.title('Métricas por limiar — sistema completo')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.ylim(50, 102)

plt.subplot(1, 2, 2)
plt.plot(df_limiares['limiar']*100, df_limiares['f1'],
         color='#7F77DD', linewidth=2.5)
plt.axvline(x=melhor_limiar_sistema*100, color='red',
            linestyle='--', linewidth=1.5,
            label=f'Pico F1: {melhor_f1_sistema*100:.2f}% no limiar {melhor_limiar_sistema*100:.0f}%')
plt.axvline(x=65, color='gray', linestyle=':',
            linewidth=1, label='Limiar atual (65%)')
plt.xlabel('Limiar de confiança (%)')
plt.ylabel('F1-Score (%)')
plt.title('F1-Score por limiar (zoom)')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('limiar_ideal.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("RESUMO FINAL")
print("="*60)
print(f"\nLimiar atual no código:  65%")
print(f"Limiar ideal calculado:  {melhor_limiar_sistema*100:.0f}%")

if melhor_limiar_sistema > 0.65:
    print(f"\nRecomendação: AUMENTAR para {melhor_limiar_sistema*100:.0f}%")
elif melhor_limiar_sistema < 0.65:
    print(f"\nRecomendação: DIMINUIR para {melhor_limiar_sistema*100:.0f}%")
else:
    print(f"\nLimiar atual de 65% já é o ideal!")

print(f"\nPara aplicar: substitua LIMIAR = 0.65 por LIMIAR = {melhor_limiar_sistema:.2f}")
print("nos arquivos app_treino.py")
print("="*60)