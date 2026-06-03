import pickle
from datetime import datetime
import os

with open("modelos.pkl", "rb") as f:
    dados = pickle.load(f)

# Verifica data de modificação do arquivo
data_modificacao = os.path.getmtime("modelos.pkl")
data_formatada = datetime.fromtimestamp(data_modificacao).strftime("%d/%m/%Y às %H:%M")

print("=" * 50)
print("INFORMAÇÕES DO MODELO ATUAL")
print("=" * 50)
print(f"Última modificação: {data_formatada}")
print(f"Modelos salvos: {list(dados['modelos'].keys())}")
print(f"Melhor modelo: {dados['melhor_nome']}")

if "total_noticias" in dados:
    print(f"Total de notícias originais: {dados['total_noticias']}")

if "X_test" in dados:
    print(f"Conjunto de teste salvo: {dados['X_test'].shape[0]} notícias")

# Verifica feedback
if os.path.exists("feedback.csv"):
    import pandas as pd
    df = pd.read_csv("feedback.csv")
    print(f"\nFeedbacks salvos: {len(df)} exemplos")
    print(f"  Fake: {sum(df['label_correto']==0)}")
    print(f"  Verdadeiras: {sum(df['label_correto']==1)}")
else:
    print("\nNenhum feedback salvo ainda.")

print("=" * 50)