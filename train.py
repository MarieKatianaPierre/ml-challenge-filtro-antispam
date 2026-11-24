import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data/spam.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
GRAFICOS_DIR = Path("graficos")


def carregar_dataset(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo '{caminho}' não encontrado. Coloque o dataset CSV em data/spam.csv."
        )

    df = None
    for sep in [",", "\t", ";"]:
        try:
            temp = pd.read_csv(caminho, sep=sep, encoding="utf-8")
            if temp.shape[1] >= 2:
                df = temp
                break
        except Exception:
            continue

    if df is None:
        df = pd.read_csv(caminho, encoding="latin-1")

    col_map = {}
    if {"v1", "v2"}.issubset(df.columns):
        col_map = {"v1": "label", "v2": "message"}
    elif {"Category", "Message"}.issubset(df.columns):
        col_map = {"Category": "label", "Message": "message"}
    elif {"text", "label"}.issubset(df.columns):
        col_map = {"label": "label", "text": "message"}

    if col_map:
        df = df.rename(columns=col_map)

    if not {"label", "message"}.issubset(df.columns):
        raise ValueError(
            f"Colunas esperadas ('label' e 'message') não encontradas. Colunas atuais: {list(df.columns)}"
        )

    df = df[["label", "message"]].dropna().copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"] != ""].drop_duplicates(subset=["message"]).reset_index(drop=True)

    substituicoes = {
        "0": "ham",
        "1": "spam",
        "legitima": "ham",
        "legítima": "ham",
        "normal": "ham",
    }
    df["label"] = df["label"].replace(substituicoes)
    df = df[df["label"].isin(["ham", "spam"])].reset_index(drop=True)
    return df


def gerar_analise_exploratoria(df: pd.DataFrame) -> dict:
    GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)
    df_eda = df.copy()
    df_eda["num_chars"] = df_eda["message"].str.len()
    df_eda["num_palavras"] = df_eda["message"].str.split().str.len()

    contagem = df_eda["label"].value_counts()
    proporcao = df_eda["label"].value_counts(normalize=True) * 100

    plt.figure(figsize=(6, 4))
    cores = ["#2b8cbe", "#d95f02"]
    ax = contagem.reindex(["ham", "spam"]).plot(kind="bar", color=cores)
    plt.title("Distribuição de Mensagens Legítimas (ham) vs. Spam")
    plt.xlabel("Classe")
    plt.ylabel("Quantidade de mensagens")
    plt.xticks(ticks=[0, 1], labels=["Legítima (ham)", "Spam"], rotation=0)
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / "distribuicao_classes.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    ham_chars = df_eda[df_eda["label"] == "ham"]["num_chars"]
    spam_chars = df_eda[df_eda["label"] == "spam"]["num_chars"]
    plt.hist(ham_chars, bins=35, alpha=0.65, label="Legítima (ham)", color="#2b8cbe")
    plt.hist(spam_chars, bins=35, alpha=0.65, label="Spam", color="#d95f02")
    plt.title("Distribuição do Tamanho das Mensagens (Caracteres)")
    plt.xlabel("Quantidade de caracteres")
    plt.ylabel("Frequência")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / "tamanho_mensagens.png", dpi=150)
    plt.close()

    resumo = {
        "total_mensagens": int(len(df_eda)),
        "qtd_ham": int(contagem.get("ham", 0)),
        "qtd_spam": int(contagem.get("spam", 0)),
        "pct_spam": round(float(proporcao.get("spam", 0.0)), 2),
        "media_chars_ham": round(float(ham_chars.mean()), 1),
        "media_chars_spam": round(float(spam_chars.mean()), 1),
        "media_palavras_ham": round(float(df_eda[df_eda["label"] == "ham"]["num_palavras"].mean()), 1),
        "media_palavras_spam": round(float(df_eda[df_eda["label"] == "spam"]["num_palavras"].mean()), 1),
    }

    print("\n--- RESUMO EXPLORATÓRIO (EDA) ---")
    print(f"Total de mensagens após limpeza: {resumo['total_mensagens']}")
    print(f"Legítimas (ham): {resumo['qtd_ham']} | Spam: {resumo['qtd_spam']} ({resumo['pct_spam']}% spam)")
    print(f"Média de caracteres -> ham: {resumo['media_chars_ham']} | spam: {resumo['media_chars_spam']}")
    print(f"Média de palavras   -> ham: {resumo['media_palavras_ham']} | spam: {resumo['media_palavras_spam']}")
    return resumo


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = carregar_dataset(DATA_PATH)
    resumo_eda = gerar_analise_exploratoria(df)

    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelos = {
        "LogisticRegression + TF-IDF": Pipeline(
            [
                ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=5000)),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "MultinomialNB + TF-IDF": Pipeline(
            [
                ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=5000)),
                ("clf", MultinomialNB(alpha=0.5)),
            ]
        ),
    }

    resultados = []
    pipelines_treinados = {}

    print("\n--- AVALIAÇÃO NO CONJUNTO DE TESTE ---")
    for nome, pipe in modelos.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec_spam = precision_score(y_test, y_pred, pos_label="spam", zero_division=0)
        rec_spam = recall_score(y_test, y_pred, pos_label="spam", zero_division=0)
        f1_spam = f1_score(y_test, y_pred, pos_label="spam", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"]).tolist()
        relatorio = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        print(f"\nModelo: {nome}")
        print(f"Acurácia: {acc:.4f} | Precisão (spam): {prec_spam:.4f} | Recall (spam): {rec_spam:.4f} | F1 (spam): {f1_spam:.4f}")
        print(f"Matriz de confusão [linhas=real (ham, spam), colunas=previsto (ham, spam)]: {cm}")

        resultados.append(
            {
                "algoritmo": nome,
                "acuracia": round(float(acc), 4),
                "precisao_spam": round(float(prec_spam), 4),
                "recall_spam": round(float(rec_spam), 4),
                "f1_spam": round(float(f1_spam), 4),
                "f1_macro": round(float(f1_macro), 4),
                "matriz_confusao_ham_spam": cm,
                "relatorio_classes": relatorio,
            }
        )
        pipelines_treinados[nome] = pipe

    melhor_res = max(resultados, key=lambda item: item["f1_spam"])
    melhor_nome = melhor_res["algoritmo"]
    melhor_pipe = pipelines_treinados[melhor_nome]

    caminho_modelo = MODELS_DIR / "model.joblib"
    joblib.dump(melhor_pipe, caminho_modelo)

    metricas_payload = {
        "eda": resumo_eda,
        "modelo_selecionado": melhor_nome,
        "criterio_selecao": "Maior F1-score na classe spam no conjunto de teste",
        "modelos": resultados,
    }

    caminho_metricas = REPORTS_DIR / "metricas.json"
    with open(caminho_metricas, "w", encoding="utf-8") as f:
        json.dump(metricas_payload, f, ensure_ascii=False, indent=2)

    print(f"\nModelo selecionado e salvo em: {caminho_modelo} -> {melhor_nome}")
    print(f"Métricas salvas em: {caminho_metricas}")


if __name__ == "__main__":
    main()
