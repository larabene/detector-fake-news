# ================================================
# 1_treino.py — Treina com FakeBR + FakeRecogna
# Versão melhorada — correções aplicadas:
# - cv=5 no SVM (corrige nan% de confiança)
# - min_df=2 no TF-IDF (reduz ruído)
# - n_estimators=300 no Random Forest (mais estável)
# - y_train salvo no modelos.pkl
# ================================================

import pandas as pd
import numpy as np
import re
import unicodedata
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords

# ================================================
# PASSO 1 — Baixar os dois datasets
# ================================================
print("=" * 50)
print("CARREGANDO OS DATASETS")
print("=" * 50)

# Dataset 1 — FakeRecogna
print("\n[1/2] Baixando FakeRecogna...")
url_fakerecogna = "https://huggingface.co/datasets/recogna-nlp/FakeRecogna/resolve/main/FakeRecogna.csv"
df_recogna = pd.read_csv(url_fakerecogna)
df_recogna = df_recogna[['Noticia', 'Classe']].copy()
df_recogna = df_recogna.dropna()
df_recogna.columns = ['texto', 'label']
df_recogna['label'] = df_recogna['label'].astype(int)
df_recogna['fonte'] = 'FakeRecogna'
print(f"✓ FakeRecogna carregado: {len(df_recogna)} notícias")
print(f"   Fake: {sum(df_recogna['label']==0)} | Verdadeiras: {sum(df_recogna['label']==1)}")

# Dataset 2 — FakeBR
print("\n[2/2] Carregando FakeBR...")
df_fakebr = pd.read_csv("pre-processed.csv")
df_fakebr = df_fakebr[['preprocessed_news', 'label']].copy()
df_fakebr = df_fakebr.dropna()
df_fakebr.columns = ['texto', 'label']
df_fakebr['label'] = df_fakebr['label'].map({'fake': 0, 'true': 1})
df_fakebr = df_fakebr.dropna()
df_fakebr['label'] = df_fakebr['label'].astype(int)
df_fakebr['fonte'] = 'FakeBR'
print(f"✓ FakeBR carregado: {len(df_fakebr)} notícias")
print(f"   Fake: {sum(df_fakebr['label']==0)} | Verdadeiras: {sum(df_fakebr['label']==1)}")

# ================================================
# PASSO 2 — Combinar os dois datasets
# ================================================
print("\n" + "=" * 50)
print("COMBINANDO OS DATASETS")
print("=" * 50)

dados = pd.concat([df_recogna, df_fakebr], ignore_index=True)
dados = dados.dropna(subset=['texto', 'label'])
dados = dados.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n✓ Dataset combinado!")
print(f"   Total: {len(dados)} notícias")
print(f"   Fake: {sum(dados['label']==0)}")
print(f"   Verdadeiras: {sum(dados['label']==1)}")
print(f"   FakeRecogna: {sum(dados['fonte']=='FakeRecogna')} notícias")
print(f"   FakeBR: {sum(dados['fonte']=='FakeBR')} notícias")

# ================================================
# PASSO 3 — Pré-processar
# ================================================
print("\n" + "=" * 50)
print("PRÉ-PROCESSANDO")
print("=" * 50)

def preprocessar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+', '', texto)
    texto = re.sub(r'[^a-záéíóúâêîôûãõàèìòùç\s]', '', texto)
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    tokens = texto.split()
    tokens = [t for t in tokens if len(t) > 2]
    return ' '.join(tokens)

def padronizar_fakebr(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    tokens = texto.split()
    tokens = [t for t in tokens if len(t) > 1]
    return ' '.join(tokens)

print("Aplicando pré-processamento...")

mask_recogna = dados['fonte'] == 'FakeRecogna'
mask_fakebr  = dados['fonte'] == 'FakeBR'
dados.loc[mask_recogna, 'texto_limpo'] = dados.loc[mask_recogna, 'texto'].apply(preprocessar_texto)
dados.loc[mask_fakebr,  'texto_limpo'] = dados.loc[mask_fakebr,  'texto'].apply(padronizar_fakebr)

dados = dados[dados['texto_limpo'].str.strip() != '']
dados = dados.reset_index(drop=True)

print(f"✓ Pré-processamento concluído!")
print(f"   FakeRecogna: limpeza completa aplicada")
print(f"   FakeBR: apenas padronização aplicada")
print(f"   Total após limpeza: {len(dados)} notícias")

# ================================================
# PASSO 4 — Vetorizar e dividir
# ================================================
print("\n" + "=" * 50)
print("VETORIZANDO COM TF-IDF")
print("=" * 50)

# MELHORIA: min_df=2 descarta palavras que aparecem em menos de
# 2 documentos — reduz ruído e melhora o Naive Bayes
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2               # novidade
)

X = vectorizer.fit_transform(dados['texto_limpo'])
y = dados['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Treino: {X_train.shape[0]} notícias")
print(f"✓ Teste:  {X_test.shape[0]} notícias")

# ================================================
# PASSO 5 — Treinar os 4 modelos
# ================================================
print("\n" + "=" * 50)
print("TREINANDO OS 4 MODELOS")
print("(O Random Forest pode demorar 5-10 minutos)")
print("=" * 50)

modelos = {
    'Naive Bayes':    MultinomialNB(),

    'Reg. Logística': LogisticRegression(
                          max_iter=1000,
                          random_state=42,
                          class_weight='balanced'
                      ),

    # MELHORIA: cv=5 garante calibração robusta e corrige nan%
    'SVM':            CalibratedClassifierCV(
                          LinearSVC(
                              random_state=42,
                              class_weight='balanced'
                          ),
                          cv=5                   # novidade
                      ),

    # MELHORIA: 300 árvores — resultado mais estável
    'Random Forest':  RandomForestClassifier(
                          n_estimators=300,      # era 200
                          random_state=42,
                          class_weight='balanced',
                          n_jobs=-1
                      ),
}

resultados = []
modelos_treinados = {}

for nome, modelo in modelos.items():
    print(f"\n  Treinando {nome}...")
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    resultados.append({
        'Modelo':   nome,
        'Acurácia': round(accuracy_score(y_test, y_pred)  * 100, 2),
        'Precisão': round(precision_score(y_test, y_pred) * 100, 2),
        'Recall':   round(recall_score(y_test, y_pred)    * 100, 2),
        'F1-Score': round(f1_score(y_test, y_pred)        * 100, 2),
    })
    modelos_treinados[nome] = (modelo, y_pred)
    print(f"  ✓ {nome} concluído!")

df_resultados = pd.DataFrame(resultados)
melhor_nome = df_resultados.loc[df_resultados['F1-Score'].idxmax(), 'Modelo']

print("\n" + "=" * 50)
print("COMPARAÇÃO DOS 4 MODELOS")
print("=" * 50)
print(df_resultados.to_string(index=False))
print(f"\nMelhor modelo: {melhor_nome}")

# ================================================
# PASSO 6 — Salvar tudo incluindo conjunto de teste
# ================================================
print("\n" + "=" * 50)
print("SALVANDO MODELOS")
print("=" * 50)

with open("modelos.pkl", "wb") as f:
    pickle.dump({
        "modelos":         modelos_treinados,
        "vectorizer":      vectorizer,
        "melhor_nome":     melhor_nome,
        "resultados":      df_resultados,
        "datasets_usados": ["FakeRecogna", "FakeBR"],
        "total_noticias":  len(dados),
        "X_test":          X_test,
        "y_test":          y_test,
        "y_train":         y_train,   # novidade
    }, f)

print("✓ modelos.pkl salvo com sucesso!")
print("\n" + "=" * 50)
print("TREINAMENTO CONCLUÍDO!")
print(f"Total de notícias usadas: {len(dados)}")
print(f"Melhor modelo: {melhor_nome}")
print("\nPróximos passos:")
print("  Calibrar:  streamlit run app_treino.py")
print("  Público:   streamlit run app_publico.py")
print("  Otimizar:  python otimizar_limiar.py")
print("=" * 50)