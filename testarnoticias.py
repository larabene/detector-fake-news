import pickle
import re
import unicodedata

with open("modelos.pkl", "rb") as f:
    dados = pickle.load(f)

modelos_treinados = dados["modelos"]
vectorizer = dados["vectorizer"]

pesos = {
    "Naive Bayes":    1,
    "Reg. Logística": 2,
    "SVM":            4,
    "Random Forest":  2
}

def preprocessar(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"[^a-z\s]", "", texto)
    texto = "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")
    return " ".join([t for t in texto.split() if len(t) > 2])

noticias = {
    "Notícia política verdadeira (STF)": "o supremo tribunal federal decidiu por unanimidade manter a decisao sobre as emendas parlamentares segundo informou o stf nesta terca feira",
    "Notícia econômica verdadeira (salário mínimo)": "o presidente sancionou a lei que aumenta o salario minimo segundo informou o ministerio da fazenda nesta terca feira",
    "Fake news política óbvia": "dilma e lula planejaram golpe contra o brasil segundo portal diario brasil o petista seria ditador",
    "Fake news conspiratória (microchip/vacina)": "medico expoe verdade que governo tenta esconder vacina causa microchip nos brasileiros compartilhe antes que censurem",
    "Notícia de saúde verdadeira (Fiocruz)": "a fiocruz informou nesta terca feira que pesquisadores desenvolveram novo tratamento para dengue segundo comunicado oficial da instituicao"
}

print("=" * 70)
for descricao, texto in noticias.items():
    texto_limpo = preprocessar(texto)
    vetor = vectorizer.transform([texto_limpo])
    
    print(f"\nNOTÍCIA: {descricao}")
    print(f"Texto processado: {texto_limpo[:80]}...")
    print(f"{'Modelo':<20} {'Voto':<12} {'Confiança':<12} {'Peso':<6} {'Pontos'}")
    print("-" * 60)
    
    votos_fake = 0
    votos_true = 0
    
    for nome, (modelo, _) in modelos_treinados.items():
        pred = modelo.predict(vetor)[0]
        prob = modelo.predict_proba(vetor)[0]
        confianca = round(max(prob) * 100, 1)
        peso = pesos[nome]
        voto = "VERDADEIRA" if pred == 1 else "FAKE NEWS"
        pontos = peso if pred == 1 else 0
        
        if pred == 1:
            votos_true += peso
        else:
            votos_fake += peso
            
        print(f"{nome:<20} {voto:<12} {confianca}%{'':<6} {peso}x{'':<4} {pontos} pts")
    
    decisao = "VERDADEIRA" if votos_true > votos_fake else "FAKE NEWS"
    print("-" * 60)
    print(f"PLACAR: {votos_true} pts verdadeira x {votos_fake} pts fake")
    print(f"DECISÃO FINAL: {decisao}")
    print("=" * 70)
    