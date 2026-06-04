import streamlit as st
import pickle
import re
import unicodedata
import numpy as np

st.set_page_config(
    page_title="Detector de Fake News",
    page_icon="🔍",
    layout="centered"
)

@st.cache_resource
def carregar_modelos():
    with open("modelos.pkl", "rb") as f:
        return pickle.load(f)

dados_salvos = carregar_modelos()
modelos_treinados = dados_salvos["modelos"]
vectorizer = dados_salvos["vectorizer"]
melhor_nome = dados_salvos["melhor_nome"]

pesos = {
    "Naive Bayes":    1,
    "Reg. Logística": 2,
    "SVM":            4,
    "Random Forest":  2
}

LIMIAR = 0.58

def preprocessar_novo(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"[^a-záéíóúâêîôûãõàèìòùç\s]", "", texto)
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    tokens = texto.split()
    tokens = [t for t in tokens if len(t) > 2]
    return " ".join(tokens)

def classificar(texto):
    texto_limpo = preprocessar_novo(texto)
    vetor = vectorizer.transform([texto_limpo])
    votos_verdadeira = 0
    votos_fake = 0
    detalhes = []
    for nome, (modelo, _) in modelos_treinados.items():
        pred_original = modelo.predict(vetor)[0]
        prob = modelo.predict_proba(vetor)[0]
        confianca_raw = max(prob)
        confianca = round(confianca_raw * 100, 1)
        peso = pesos[nome]
        if pred_original == 1 and confianca_raw < LIMIAR:
            pred = 0
        else:
            pred = pred_original
        if pred == 1:
            votos_verdadeira += peso
        else:
            votos_fake += peso
        detalhes.append({
            "modelo": nome,
            "resultado": "✓ VERDADEIRA" if pred == 1 else "✗ FAKE NEWS",
            "confianca": confianca,
            "peso": peso,
            "pred": pred,
            "prob_fake": round(prob[0] * 100, 1),
            "prob_true": round(prob[1] * 100, 1),
        })
    total = votos_verdadeira + votos_fake
    decisao = "VERDADEIRA" if votos_verdadeira > votos_fake else "FAKE NEWS"
    confianca_sistema = round(
        (votos_verdadeira if decisao == "VERDADEIRA" else votos_fake) / total * 100, 1
    )
    return decisao, confianca_sistema, detalhes, votos_verdadeira, votos_fake

# ==================== INTERFACE ====================
st.markdown("# 🔍 Detector de Fake News")
st.markdown("### Desenvolvido por Lara Tavares Boldorini Benedito")
st.markdown("*Trabalho de Conclusão de Curso — Engenharia Mecatrônica — Mackenzie 2026*")
st.divider()

with st.expander("ℹ️ Como o sistema funciona?"):
    st.markdown("""
    Este sistema utiliza **4 algoritmos de inteligência artificial** treinados com
    19.101 notícias em português brasileiro para classificar notícias como
    verdadeiras ou falsas.

    **Os 4 algoritmos utilizados e seus pesos na votação:**
    - 🟣 **Naive Bayes** — peso 1x — Acurácia: 86,71%
    - 🔵 **Regressão Logística** — peso 2x — Acurácia: 93,77%
    - 🟢 **SVM** — peso 4x — Acurácia: 94,61% *(melhor modelo)*
    - 🟡 **Random Forest** — peso 2x — Acurácia: 93,51%

    **Como a decisão é tomada:** cada algoritmo analisa a notícia e vota.
    Os votos são somados com pesos diferentes — algoritmos mais precisos
    têm mais influência. O lado que acumular mais pontos dos 9 possíveis
    é a decisão final.
    """)

st.markdown("### 📝 Cole o texto da notícia abaixo:")
texto_usuario = st.text_area(
    label="Notícia",
    placeholder="Cole aqui o texto completo da notícia que deseja verificar...",
    height=200,
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analisar = st.button("🔍 ANALISAR NOTÍCIA", use_container_width=True, type="primary")

if analisar:
    if not texto_usuario.strip():
        st.warning("⚠️ Por favor, cole o texto de uma notícia antes de analisar.")
    elif len(texto_usuario.strip().split()) < 20:
        st.warning("⚠️ Notícias muito curtas ou com poucas informações podem ser classificadas incorretamente, pois o sistema analisa padrões linguísticos que geralmente estão presentes em textos mais completos. Para melhores resultados, insira o texto integral da notícia.")
    else:
        with st.spinner("Analisando notícia..."):
            decisao, confianca_sistema, detalhes, vv, vf = classificar(texto_usuario)

        st.session_state["texto_atual"] = texto_usuario
        st.session_state["detalhes_atual"] = detalhes
        st.session_state["vv"] = vv
        st.session_state["vf"] = vf
        st.session_state["decisao_atual"] = decisao

        st.divider()

        if decisao == "VERDADEIRA":
            st.success(f"## ✅ NOTÍCIA VERDADEIRA")
            st.markdown(f"**Confiança do sistema: {confianca_sistema}%**")
            st.markdown(f"Placar ponderado: **{vv} pontos verdadeira** × {vf} pontos fake")
        else:
            st.error(f"## ❌ FAKE NEWS DETECTADA")
            st.markdown(f"**Confiança do sistema: {confianca_sistema}%**")
            st.markdown(f"Placar ponderado: {vv} pontos verdadeira × **{vf} pontos fake**")

        st.divider()

        st.markdown("### 📊 Resultado de cada modelo:")
        cols = st.columns(4)
        icones = {
            "Naive Bayes":    "🟣",
            "Reg. Logística": "🔵",
            "SVM":            "🟢",
            "Random Forest":  "🟡"
        }
        for i, d in enumerate(detalhes):
            with cols[i]:
                emoji = icones[d["modelo"]]
                st.metric(
                    label=f"{emoji} {d['modelo']} ({d['peso']}x)",
                    value="VERDADEIRA" if "VERDADEIRA" in d["resultado"] else "FAKE NEWS",
                    delta=f"Confiança: {d['confianca']}%"
                )

        st.divider()
        st.caption("⚠️ Este sistema é uma ferramenta de auxílio e pode cometer erros. Sempre verifique a notícia em fontes confiáveis.")

# ==================== ABA DE RACIOCÍNIO ====================
st.divider()

with st.expander("🔬 Ver raciocínio detalhado dos modelos", expanded=False):
    if "detalhes_atual" not in st.session_state:
        st.info("Analise uma notícia acima para ver o raciocínio de cada modelo.")
    else:
        detalhes = st.session_state["detalhes_atual"]
        vv = st.session_state["vv"]
        vf = st.session_state["vf"]
        decisao_final = st.session_state["decisao_atual"]
        texto_analisado = st.session_state["texto_atual"]

        st.markdown("#### Como cada modelo chegou à sua decisão")
        st.caption("Cada modelo analisa o texto de forma independente e emite um voto. O resultado final é a soma ponderada dos pontos.")
        st.divider()

        descricoes = {
            "Naive Bayes": "Conta quantas vezes cada palavra aparece em notícias falsas e verdadeiras. Multiplica as probabilidades e escolhe a mais provável. É o mais simples dos quatro.",
            "Reg. Logística": "Aprende um peso para cada palavra — positivo se indica verdadeira, negativo se indica fake. Soma os pesos e gera uma probabilidade. É o modelo mais transparente.",
            "SVM": "Encontra a linha que melhor separa notícias falsas de verdadeiras. Quanto mais longe do limite, maior a confiança. É o modelo com melhor desempenho (94,59% de F1-score).",
            "Random Forest": "Cria 200 árvores de decisão independentes e combina os votos de todas elas. A maioria decide. É o mais robusto a variações no texto."
        }

        icones = {
            "Naive Bayes":    "🟣",
            "Reg. Logística": "🔵",
            "SVM":            "🟢",
            "Random Forest":  "🟡"
        }

        for d in detalhes:
            nome = d["modelo"]
            is_fake = "FAKE" in d["resultado"]
            destaque = nome == melhor_nome

            col_info, col_barra = st.columns([1, 1])

            with col_info:
                titulo = f"**{icones[nome]} {nome}** (peso {d['peso']}x)"
                if destaque:
                    titulo += " ⭐ melhor modelo"
                st.markdown(titulo)
                st.caption(descricoes[nome])
                if is_fake:
                    st.error(f"✗ FAKE NEWS — {d['confianca']}% de confiança — 0 pontos")
                else:
                    st.success(f"✓ VERDADEIRA — {d['confianca']}% de confiança — {d['peso']} pontos")

            with col_barra:
                st.markdown(f"❌ Fake: **{d['prob_fake']}%**")
                st.progress(int(d['prob_fake']) if not np.isnan(d['prob_fake']) else 50)
                st.markdown(f"✅ Verdadeira: **{d['prob_true']}%**")
                st.progress(int(d['prob_true']) if not np.isnan(d['prob_true']) else 50)

            st.divider()

        # Palavras relevantes
        texto_limpo = preprocessar_novo(texto_analisado)
        vetor = vectorizer.transform([texto_limpo])
        feature_names = vectorizer.get_feature_names_out()
        indices_texto = vetor.nonzero()[1]
        modelo_lr = modelos_treinados["Reg. Logística"][0]
        coefs = modelo_lr.coef_[0]

        palavras_rel = [(feature_names[idx], coefs[idx])
                        for idx in indices_texto if abs(coefs[idx]) > 0.05]
        palavras_rel.sort(key=lambda x: abs(x[1]), reverse=True)
        top = palavras_rel[:8]

        if top:
            st.markdown("#### 🔍 Palavras que mais influenciaram a decisão")
            st.caption("Identificadas pela Regressão Logística — o modelo mais transparente do sistema.")

            col_f, col_v = st.columns(2)
            with col_f:
                st.markdown("**❌ Indicam FAKE NEWS:**")
                fake_words = [(p, s) for p, s in top if s < 0]
                if fake_words:
                    for p, s in fake_words[:4]:
                        forca = "forte" if abs(s) > 0.6 else "moderado" if abs(s) > 0.3 else "fraco"
                        st.markdown(f"- `{p}` — indicador {forca}")
                else:
                    st.caption("Nenhuma encontrada")

            with col_v:
                st.markdown("**✅ Indicam VERDADEIRA:**")
                true_words = [(p, s) for p, s in top if s > 0]
                if true_words:
                    for p, s in true_words[:4]:
                        forca = "forte" if abs(s) > 0.6 else "moderado" if abs(s) > 0.3 else "fraco"
                        st.markdown(f"- `{p}` — indicador {forca}")
                else:
                    st.caption("Nenhuma encontrada")

            st.divider()

        # Placar final
        st.markdown("#### ⚖️ Placar final da votação")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pontos VERDADEIRA", vv)
        with col2:
            st.metric("Pontos FAKE", vf)
        with col3:
            st.metric("Total de pontos", vv + vf)

        if decisao_final == "VERDADEIRA":
            st.success(f"✅ Decisão final: VERDADEIRA — {vv} pontos vs {vf} pontos fake")
        else:
            st.error(f"❌ Decisão final: FAKE NEWS — {vf} pontos vs {vv} pontos verdadeira")

        st.caption("O total sempre soma 9: SVM (4pts) + Reg. Logística (2pts) + Random Forest (2pts) + Naive Bayes (1pt).")