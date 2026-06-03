import streamlit as st
import pickle
import re
import unicodedata
import pandas as pd
import os
from datetime import datetime
from sklearn.metrics import accuracy_score
import numpy as np

st.set_page_config(
    page_title="Painel de Treinamento — Fake News",
    page_icon="🧠",
    layout="wide"
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

def salvar_exemplo(texto, label_correto):
    arquivo = "feedback.csv"
    novo = pd.DataFrame([{
        "data": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "texto": texto,
        "label_correto": label_correto
    }])
    if os.path.exists(arquivo):
        df = pd.read_csv(arquivo)
        df = pd.concat([df, novo], ignore_index=True)
    else:
        df = novo
    df.to_csv(arquivo, index=False)
    return len(df)

def retreinar_agora():
    arquivo = "feedback.csv"
    if not os.path.exists(arquivo):
        return None, 0
    df_feedback = pd.read_csv(arquivo)
    if len(df_feedback) < 2:
        return None, len(df_feedback)
    textos = [preprocessar_novo(t) for t in df_feedback["texto"].tolist()]
    labels = df_feedback["label_correto"].tolist()
    X_novo = vectorizer.transform(textos)
    y_novo = np.array(labels)
    for nome, (modelo, _) in modelos_treinados.items():
        try:
            modelo.fit(X_novo, y_novo)
            modelos_treinados[nome] = (modelo, [])
        except:
            pass
    acertos = sum(
        1 for texto, label in zip(textos, labels)
        if modelos_treinados["SVM"][0].predict(vectorizer.transform([texto]))[0] == label
    )
    acuracia = round(acertos / len(labels) * 100, 1)
    with open("modelos.pkl", "wb") as f:
        pickle.dump({
            "modelos": modelos_treinados,
            "vectorizer": vectorizer,
            "melhor_nome": melhor_nome,
            "resultados": dados_salvos.get("resultados")
        }, f)
    return acuracia, len(df_feedback)

# ==================== INTERFACE ====================
st.markdown("# 🧠 Painel de Treinamento")
st.markdown("*Calibre o modelo antes de publicar para os usuários*")
st.divider()

col_m1, col_m2, col_m3 = st.columns(3)
total_feedbacks = 0
fake_count = 0
true_count = 0
if os.path.exists("feedback.csv"):
    df_fb = pd.read_csv("feedback.csv")
    total_feedbacks = len(df_fb)
    fake_count = sum(df_fb["label_correto"] == 0)
    true_count = sum(df_fb["label_correto"] == 1)

with col_m1:
    st.metric("Total de exemplos adicionados", total_feedbacks)
with col_m2:
    st.metric("Exemplos FAKE adicionados", fake_count)
with col_m3:
    st.metric("Exemplos VERDADEIROS adicionados", true_count)

st.divider()

col_esq, col_dir = st.columns([1.2, 1])

with col_esq:
    st.markdown("### 📝 Testar e adicionar exemplo")
    texto_usuario = st.text_area(
        "Cole o texto da notícia:",
        placeholder="Cole aqui o texto completo da notícia...",
        height=180
    )
    if st.button("🔍 Analisar", use_container_width=True, type="primary"):
        if texto_usuario.strip() and len(texto_usuario.split()) >= 10:
            decisao, confianca, detalhes, vv, vf = classificar(texto_usuario)
            st.session_state["texto_atual"] = texto_usuario
            st.session_state["decisao_atual"] = decisao
            st.session_state["detalhes_atual"] = detalhes
            st.session_state["vv"] = vv
            st.session_state["vf"] = vf
            st.session_state["confianca_sistema"] = confianca
            if decisao == "VERDADEIRA":
                st.success(f"✅ VERDADEIRA — Confiança: {confianca}%")
            else:
                st.error(f"❌ FAKE NEWS — Confiança: {confianca}%")
            st.markdown("**Resultado de cada modelo:**")
            for d in detalhes:
                emoji = "✅" if "VERDADEIRA" in d["resultado"] else "❌"
                st.markdown(f"- {d['modelo']} (peso {d['peso']}x): {emoji} {d['confianca']}%")
        else:
            st.warning("Cole um texto com pelo menos 10 palavras.")

    st.divider()
    st.markdown("### ✏️ Adicionar exemplo manualmente")
    st.markdown("*Use para adicionar notícias que você já sabe a classificação correta*")
    texto_manual = st.text_area(
        "Texto da notícia:",
        placeholder="Cole aqui o texto...",
        height=120,
        key="manual"
    )
    col_fake, col_true = st.columns(2)
    with col_fake:
        if st.button("➕ Adicionar como FAKE", use_container_width=True):
            if texto_manual.strip():
                total = salvar_exemplo(texto_manual, 0)
                st.success(f"✅ Salvo como FAKE! Total: {total} exemplos")
                st.rerun()
            else:
                st.warning("Cole o texto primeiro!")
    with col_true:
        if st.button("➕ Adicionar como VERDADEIRA", use_container_width=True):
            if texto_manual.strip():
                total = salvar_exemplo(texto_manual, 1)
                st.success(f"✅ Salvo como VERDADEIRA! Total: {total} exemplos")
                st.rerun()
            else:
                st.warning("Cole o texto primeiro!")

with col_dir:
    st.markdown("### 🎯 Corrigir resultado do teste")
    if "texto_atual" in st.session_state:
        st.info(f"Último resultado: **{st.session_state['decisao_atual']}**")
        st.markdown("O sistema errou? Corrija abaixo:")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("✅ Era VERDADEIRA", use_container_width=True):
                total = salvar_exemplo(st.session_state["texto_atual"], 1)
                st.success(f"Salvo! Total: {total}")
                st.rerun()
        with col_c2:
            if st.button("❌ Era FAKE NEWS", use_container_width=True):
                total = salvar_exemplo(st.session_state["texto_atual"], 0)
                st.success(f"Salvo! Total: {total}")
                st.rerun()
    else:
        st.info("Analise uma notícia ao lado para corrigir o resultado aqui.")

    st.divider()
    st.markdown("### 🔄 Retreinar o modelo")
    st.markdown(f"Você tem **{total_feedbacks} exemplos** prontos para treino.")
    if total_feedbacks < 2:
        st.warning("Adicione pelo menos 2 exemplos para retreinar.")
    else:
        st.success(f"✅ Pronto para retreinar com {total_feedbacks} exemplos!")
        if st.button("🚀 RETREINAR AGORA", use_container_width=True, type="primary"):
            with st.spinner("Retreinando o modelo... aguarde."):
                acuracia, total = retreinar_agora()
            if acuracia is not None:
                st.success(f"✅ Modelo atualizado! Acurácia: {acuracia}%")
                st.balloons()
            else:
                st.error("Adicione mais exemplos e tente novamente.")

    st.divider()
    st.markdown("### 📋 Últimos exemplos adicionados")
    if os.path.exists("feedback.csv"):
        df_show = pd.read_csv("feedback.csv")
        df_show["label"] = df_show["label_correto"].map({0: "❌ FAKE", 1: "✅ VERDADEIRA"})
        df_show["texto_curto"] = df_show["texto"].str[:60] + "..."
        st.dataframe(
            df_show[["data", "label", "texto_curto"]].tail(8),
            use_container_width=True,
            hide_index=True
        )
        if st.button("🗑️ Limpar todos os exemplos", type="secondary"):
            os.remove("feedback.csv")
            st.warning("Exemplos removidos!")
            st.rerun()
    else:
        st.info("Nenhum exemplo adicionado ainda.")

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
                st.progress(int(d['prob_fake']))
                st.markdown(f"✅ Verdadeira: **{d['prob_true']}%**")
                st.progress(int(d['prob_true']))

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