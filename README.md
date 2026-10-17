# 🛡️ Prevenção de Fraude em Sinistros de Seguro
Link do App: https://prevencao-de-fraude-em-sinistros-c6vehmqz32u4eifrjlahp4.streamlit.app/

Projeto de ciência de dados **ponta a ponta**: base de dados com problemas reais, tratamento no pandas,
EDA, 3 modelos do scikit-learn com métricas de negócio e explicabilidade, modelo no MLflow,
monitoramento e um app em Streamlit que ajuda analistas a decidir **quais sinistros investigar primeiro**.

## 1. Problema de negócio
Investigar todos os sinistros é caro e impossível; investigar poucos deixa a fraude passar.
**Pergunta central:** *dentro da capacidade da equipe, quais sinistros devo investigar para proteger o maior valor, já descontando o custo de investigar?*

## 2. Resultado (base de teste, 4.000 sinistros, fila de 10%)

| Estratégia | Resultado líquido |
|---|---|
| Investigar todos | −R$ 0,48 mi |
| Investigar ao acaso | −R$ 0,05 mi |
| Maiores valores (sem modelo) | R$ 1,54 mi |
| **Modelo + valor esperado** | **R$ 2,90 mi** |

A fila de 400 casos encontra 141 das 303 fraudes (35% de precisão contra 7,6% da base). Dados fictícios e premissas ilustrativas: veja o [Model Card](docs/MODEL_CARD.md).

## 3. Objetivos e onde são atendidos

| # | Objetivo | Onde |
|---|---|---|
| 1 | Detectar sinistros suspeitos | Etapa 4, `src/04_modelagem.py` |
| 2 | Combinar segurado, apólice, acidente e histórico | Etapa 2, `src/02_preprocessamento.py` |
| 3 | Reduzir perdas financeiras (valor protegido − custo) | `src/comum.py` e Etapa 4 |
| 4 | Otimizar o trabalho dos analistas | Etapa 5, `src/05_priorizacao.py` |
| 5 | Monitorar o modelo | Etapas 6 e 7 |

## 4. Roteiro

| Etapa | O que faz | Arquivo | Documento |
|---|---|---|---|
| 1 | Gera a base sintética "suja" | `src/01_gerar_dados.py` | [01](docs/01_dicionario_de_dados.md) |
| 2 | Limpeza e variáveis no pandas | `src/02_preprocessamento.py` | [02](docs/02_preprocessamento.md) |
| 3 | EDA com 9 gráficos | `src/03_eda.py` | [03](docs/03_eda.md) |
| 4 | 3 modelos, métricas de negócio, explicabilidade | `src/04_modelagem.py` | [04](docs/04_modelagem.md) |
| 5 | Fila priorizada por analista | `src/05_priorizacao.py` | [05](docs/05_priorizacao.md) |
| 6 | Modelo empacotado no MLflow | `src/06_mlflow_registro.py` | [06](docs/06_mlflow.md) |
| 7 | Monitoramento (PSI, AUC, alertas) | `src/07_monitoramento.py` | [07](docs/07_monitoramento.md) |
| 8 | App Streamlit | `app/app.py` | [08](docs/08_app.md) |
| 9 | Testes automáticos | `tests/` | – |

Outros documentos: [Arquitetura](docs/ARQUITETURA.md) · [Model Card](docs/MODEL_CARD.md)

## 5. Como rodar

```bash
cd projeto_fraude_seguro
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python rodar_pipeline.py           # roda as etapas 1 a 7 na ordem certa (uns 2 minutos)
streamlit run app/app.py           # abre o app

pytest -q                          # testes (opcional)
mlflow ui --backend-store-uri sqlite:///mlflow.db   # tela do MLflow (opcional)
```

Se o MLflow não estiver instalado, use `python rodar_pipeline.py --sem-mlflow`.
Para recomeçar de uma etapa: `python rodar_pipeline.py --a-partir-de 4`.

## 6. Estrutura

```
projeto_fraude_seguro/
├── app/app.py               # Streamlit
├── src/                     # 01_ a 07_ + comum.py (regras compartilhadas)
├── data/raw | processed     # bases e previsões
├── models/                  # modelos .joblib e metricas.json
├── reports/                 # tabelas, monitoramento e figuras/
├── docs/                    # documentação
├── tests/                   # pytest
├── rodar_pipeline.py        # roda tudo
└── requirements.txt
```

## 7. Aviso
Todos os dados são **fictícios**, criados só para estudo. O projeto não usa senhas, chaves de API nem documentos pessoais.
