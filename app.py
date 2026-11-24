import json
from pathlib import Path
import joblib
import streamlit as st

MODEL_PATH = Path("models/model.joblib")
METRICS_PATH = Path("reports/metricas.json")

st.set_page_config(page_title="Filtro Antispam — ML Challenge", page_icon="📩", layout="centered")

st.title("📩 Filtro Antispam de Mensagens")
st.caption("Protótipo educacional de Machine Learning — as previsões são probabilísticas e sujeitas a erros.")

st.markdown(
    """
    **Sobre o problema:** esta aplicação classifica mensagens de texto como **Legítima (`ham`)** ou **Spam**,
    auxiliando na triagem automática de mensagens indesejadas com base em padrões textuais aprendidos em dados reais.
    """
)

if not MODEL_PATH.exists():
    st.error(
        "Arquivo do modelo (`models/model.joblib`) não encontrado. "
        "Execute primeiro `python train.py` no terminal para treinar e salvar o modelo."
    )
    st.stop()

@st.cache_resource
def carregar_modelo(path: Path):
    return joblib.load(path)

@st.cache_data
def carregar_metricas(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

modelo = carregar_modelo(MODEL_PATH)
metricas = carregar_metricas(METRICS_PATH)

# Exibe informações do modelo e métrica de avaliação
if metricas:
    nome_modelo = metricas.get("modelo_selecionado", "Modelo treinado")
    dados_melhor = next(
        (m for m in metricas.get("modelos", []) if m["algoritmo"] == nome_modelo),
        None,
    )
    st.info(
        f"**Algoritmo utilizado:** {nome_modelo}"
        + (
            f" | **F1-score (spam) no teste:** {dados_melhor['f1_spam']:.4f} "
            f"| **Acurácia no teste:** {dados_melhor['acuracia']:.4f}"
            if dados_melhor
            else ""
        )
    )
else:
    st.info("**Algoritmo utilizado:** Pipeline de classificação de texto salvo em `models/model.joblib`.")

texto_usuario = st.text_area(
    "Digite ou cole a mensagem para análise:",
    height=140,
    placeholder="Ex.: Parabéns! Você ganhou um prêmio exclusivo. Clique no link agora para resgatar...",
)

if st.button("Fazer previsão", type="primary"):
    mensagem_limpa = texto_usuario.strip()

    if not mensagem_limpa:
        st.warning("Por favor, digite uma mensagem válida antes de realizar a previsão.")
    elif len(mensagem_limpa) < 2:
        st.warning("A mensagem é muito curta. Digite pelo menos 2 caracteres.")
    else:
        predicao = modelo.predict([mensagem_limpa])[0]

        prob_spam = None
        if hasattr(modelo, "predict_proba"):
            classes = list(modelo.classes_)
            probs = modelo.predict_proba([mensagem_limpa])[0]
            if "spam" in classes:
                prob_spam = float(probs[classes.index("spam")])

        if predicao == "spam":
            st.error("### 🚫 Resultado: **SPAM**")
            explicacao = (
                "O modelo identificou padrões de vocabulário e estrutura mais próximos de mensagens "
                "promocionais indesejadas, ofertas suspeitas ou solicitações de ação rápida."
            )
            if prob_spam is not None:
                explicacao += f" Probabilidade estimada de spam: **{prob_spam * 100:.1f}%**."
            st.write(explicacao)
        else:
            st.success("### ✅ Resultado: **LEGÍTIMA (`ham`)**")
            explicacao = (
                "O modelo classificou a mensagem como legítima, com maior semelhança a comunicações "
                "comuns do dia a dia em relação aos exemplos de spam vistos no treino."
            )
            if prob_spam is not None:
                explicacao += f" Probabilidade estimada de spam: **{prob_spam * 100:.1f}%**."
            st.write(explicacao)

        st.caption(
            "Nota: falsos positivos podem ocultar mensagens importantes e falsos negativos podem deixar passar "
            "mensagens indesejadas; use o resultado como apoio."
        )