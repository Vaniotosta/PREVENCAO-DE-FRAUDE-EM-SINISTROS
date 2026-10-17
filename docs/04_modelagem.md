# Etapa 4 – Modelagem

Arquivo: `src/04_modelagem.py` · Funções de apoio: `src/comum.py`

## Desenho experimental

| Item | Decisão | Por quê |
|---|---|---|
| Divisão | **Temporal**: treino 60% (até 20/10/2024), validação 20% (até 26/05/2025), teste 20% (até 30/12/2025) | Em produção o modelo prevê o futuro; sorteio aleatório deixaria o resultado otimista |
| Escolha do modelo | Maior **resultado líquido na validação** | A decisão é financeira, não só estatística |
| Teste | Usado só para a nota final | Evita escolher o modelo "olhando a prova" |
| Pipeline | Imputação (mediana/moda) → padronização → one-hot → modelo → calibração | Tudo é ajustado só com o treino, sem vazamento |
| Calibração | `CalibratedClassifierCV` (sigmoid, 3 partes) | O valor esperado usa a probabilidade, que precisa ser realista |
| Variáveis | 42 (37 numéricas e 5 categóricas); `mes_sinistro` ficou de fora | Com divisão temporal, o mês só mede cobertura de calendário |

## Os 3 modelos (scikit-learn)

| Modelo | Papel |
|---|---|
| Regressão Logística | Simples, rápida e fácil de explicar |
| Random Forest | Captura interações entre variáveis |
| Gradient Boosting (`HistGradientBoostingClassifier`) | Costuma ser o mais preciso em dados tabulares |

## Métricas

**Estatísticas:** ROC-AUC (separa fraude de legítimo), PR-AUC (foco na classe rara) e Brier (qualidade da probabilidade).

**De negócio** (capacidade de investigar 10% dos sinistros; custo R$ 1.500 por caso; 70% do valor do sinistro evitado quando a fraude é confirmada):

| Métrica | Significado |
|---|---|
| Recall na capacidade | % das fraudes que a fila de 10% encontra |
| Precisão na capacidade | % da fila que é fraude de fato |
| Lift | Quantas vezes a fila é melhor que sortear |
| Valor protegido | Soma de (valor do sinistro × 70%) das fraudes encontradas |
| Custo | Casos investigados × R$ 1.500 |
| **Resultado líquido** | Valor protegido − custo |
| ROI | Resultado líquido ÷ custo |

### Resultados no teste

| Modelo | AUC | PR-AUC | Recall | Precisão | Lift | Valor protegido | Resultado líquido | ROI |
|---|---|---|---|---|---|---|---|---|
| Regressão Logística | 0,831 | 0,373 | 46,5% | 35,2% | 4,65x | R$ 3,50 mi | R$ 2,90 mi | 4,8x |
| Random Forest | 0,827 | 0,368 | 41,3% | 31,2% | 4,13x | R$ 3,28 mi | R$ 2,68 mi | 4,5x |
| Gradient Boosting | 0,837 | 0,400 | 45,5% | 34,5% | 4,55x | R$ 3,52 mi | R$ 2,92 mi | 4,9x |

Resultado líquido na validação (base da escolha): Regressão Logística R$ 2,01 mi, Gradient Boosting R$ 1,98 mi, Random Forest R$ 1,85 mi.

**Modelo escolhido: Regressão Logística.** A diferença para o Gradient Boosting é pequena e está dentro do ruído. Com desempenho parecido, o modelo mais simples é mais fácil de auditar e explicar.
O teste teve taxa de fraude maior (7,6%) que a validação (6,4%), por isso os números do teste são um pouco melhores que os da validação.

## Explicabilidade

**Global – importância por permutação** (`reports/importancia_permutacao.csv`, `modelo_05_importancia.png`): embaralha uma variável e mede quanto o PR-AUC piora. Top 5 do modelo escolhido: razão sinistro/valor segurado, apólice nova (60 dias), score de crédito, boletim de ocorrência e tipo de sinistro.

**Local – por grupos de variáveis** (`explicar_lote` em `comum.py`): troca um grupo (ex.: "Boletim de ocorrência e testemunhas") por valores típicos e mede quanto a probabilidade cai. Vale para qualquer modelo do sklearn e alimenta o "motivo principal" da fila e a aba "Analisar um sinistro" do app.

**Gráficos:** `modelo_01_roc_pr.png`, `modelo_02_resultado_liquido.png`, `modelo_03_calibracao.png`, `modelo_04_ganho_acumulado.png`, `modelo_05_importancia.png`.

## Arquivos gerados
`models/{regressao_logistica,random_forest,gradient_boosting}.joblib`, `models/modelo_final.joblib`, `models/metricas.json`, `data/processed/predicoes_teste.csv`, `reports/importancia_permutacao.csv`, `reports/curva_capacidade_teste.csv`.
