@"
# ML Challenge — Filtro Antispam de Mensagens (Trilha D)

- **Estudante:** Marie Katiana Pierre
- **Resumo do problema:** Desenvolvimento de um protótipo de Machine Learning capaz de classificar mensagens de texto como **legítimas (`ham`)** ou **spam**, auxiliando sistemas de mensageria e usuários na triagem automática de conteúdos indesejados ou suspeitos.
- **Quem utilizaria a previsão:** Provedores de comunicação, aplicativos de mensagens e usuários finais que precisam filtrar automaticamente mensagens promocionais abusivas ou tentativas de fraude antes da leitura.
- **Atributos disponíveis no momento do uso:** Apenas o texto bruto da mensagem recebida (`message`).

---

## 1. Fonte dos Dados e Variável-Alvo

- **Dataset:** SMS Spam Collection Dataset (UCI Machine Learning / Kaggle)
- **URL:** https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- **Data de acesso:** 24/11/2026
- **Licença:** Disponibilizado publicamente para fins educacionais e de pesquisa (indicada como *Unknown* na página do Kaggle).
- **Variável-alvo (`y`):** `label` (`ham` = mensagem legítima; `spam` = mensagem indesejada).

---

## 2. Análise Exploratória (EDA) e Principais Achados

Após a remoção de duplicatas e registros vazios, a base totalizou **5.158 mensagens**, sendo **4.516 legítimas (`ham`)** e **642 `spam`**.

### Três conclusões objetivas
1. **Desbalanceamento acentuado das classes:** As mensagens `spam` representam apenas **12,45%** da base (642 exemplos), enquanto as mensagens legítimas correspondem a **87,55%** (4.516 exemplos), exigindo avaliação por precisão, recall e F1-score além da acurácia.
2. **Mensagens de spam são quase duas vezes mais longas em caracteres:** A média de comprimento das mensagens `spam` é de **137,2 caracteres**, frente a **70,4 caracteres** nas mensagens `ham`, refletindo textos mais elaborados com chamadas promocionais, números de contato e instruções de resgate.
3. **Maior densidade de palavras no spam:** Em média, mensagens `spam` contêm **23,6 palavras**, contra **14,1 palavras** em mensagens legítimas.

### Visualizações geradas
- `graficos/distribuicao_classes.png`: Comparação da quantidade de mensagens legítimas (`ham`) versus `spam` após limpeza.
- `graficos/tamanho_mensagens.png`: Histograma comparativo da distribuição do número de caracteres entre mensagens legítimas e spam.

---

## 3. Pré-processamento e Prevenção de Vazamento de Dados

- Padronização dos nomes das colunas (`v1` -> `label`, `v2` -> `message`), limpeza de espaços extras e remoção de mensagens vazias ou duplicadas (evitando que mensagens idênticas apareçam simultaneamente no treino e no teste).
- Separação estratificada entre treino (**80%**) e teste (**20%**) com `random_state=42` (`train_test_split(..., stratify=y)`) **antes** de qualquer transformação textual.
- Vetorização com `TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=5000)` encapsulada dentro de um `Pipeline` do `scikit-learn`, garantindo que o vocabulário e os pesos IDF fossem ajustados exclusivamente nos dados de treino.

---

## 4. Treinamento, Comparação e Seleção do Modelo

Os modelos foram avaliados no mesmo conjunto de teste (1.032 mensagens: 904 `ham` e 128 `spam`).

| Algoritmo | Métrica principal (F1-score Spam) | Resultado obtido | Outras métricas (Teste) | Interpretação |
| :--- | :--- | :--- | :--- | :--- |
| **LogisticRegression + TF-IDF** | F1-score (`spam`) | **0,8482** | Acurácia: `0,9671` \| Precisão: `0,9896` \| Recall: `0,7422` | Alta precisão (apenas 1 falso positivo), porém deixou passar 33 mensagens de spam (recall menor). |
| **MultinomialNB + TF-IDF** | F1-score (`spam`) | **0,9068** | Acurácia: `0,9787` \| Precisão: `0,9907` \| Recall: `0,8359` | Ganho de +0,0586 no F1-score de spam em relação à Regressão Logística, reduzindo falsos negativos de 33 para 21 e mantendo apenas 1 falso positivo. |
| **Escolhido: MultinomialNB + TF-IDF** | F1-score (`spam`) | **0,9068** | Matriz de confusão: `[[903, 1], [21, 107]]` | Melhor equilíbrio entre bloquear spams (recall de 83,59%) e preservar mensagens legítimas (precisão de 99,07%). |

### Impacto de Falsos Positivos vs. Falsos Negativos
- **Falso Positivo (`ham` classificado como `spam` — 1 caso no teste):** Uma mensagem legítima importante (ex.: aviso bancário ou pessoal) é bloqueada indevidamente. O modelo selecionado manteve essa taxa extremamente baixa (903 de 904 legítimas preservadas).
- **Falso Negativo (`spam` classificado como `ham` — 21 casos no teste):** Uma mensagem indesejada chega à caixa de entrada do usuário, gerando incômodo ou risco de golpe caso contenha links maliciosos.

---

## 5. Como Instalar e Executar

```bash
# 1. Clonar o repositório e acessar a pasta
git clone [https://github.com/SEU_USUARIO/ml-challenge-filtro-antispam.git](https://github.com/SEU_USUARIO/ml-challenge-filtro-antispam.git)
cd ml-challenge-filtro-antispam

# 2. Criar e ativar o ambiente virtual
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# 3. Instalar as dependências
pip install -r requirements.txt

# 4. Treinar os modelos, gerar gráficos e salvar o artefato (opcional, já incluso)
python train.py

# 5. Executar a interface interativa Streamlit
streamlit run app.py