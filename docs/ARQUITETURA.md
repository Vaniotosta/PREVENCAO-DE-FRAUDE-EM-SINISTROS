# Arquitetura do projeto

## Fluxo de ponta a ponta

```
 01_gerar_dados ──► data/raw/sinistros_raw.csv
        │
 02_preprocessamento ──► data/processed/sinistros_tratado.csv
        │
        ├─► 03_eda ──► reports/figuras/eda_*.png, reports/eda_resumo.json
        │
 04_modelagem ──► models/*.joblib, models/modelo_final.joblib, models/metricas.json
        │         data/processed/predicoes_teste.csv, reports/importancia_permutacao.csv
        │
 05_priorizacao ──► data/processed/motivos_teste.csv, reports/fila_priorizada.csv
        │
 06_mlflow_registro ──► mlflow.db + mlruns/  (modelo_fraude_seguro@champion)
        │
 07_monitoramento ──► reports/monitoramento[_simulado].csv/.json
        │
        ▼
   app/app.py (Streamlit)  ◄── lê tudo acima, não treina nada
```

`src/comum.py` é usado por TODAS as etapas e pelo app: caminhos, variáveis, modelos, métricas de negócio, priorização, explicação e PSI. Cada regra existe em um único lugar.

## Pastas

| Pasta | Conteúdo |
|---|---|
| `src/` | Uma etapa por arquivo (`01_` a `07_`) e `comum.py` |
| `app/` | `app.py` (Streamlit) |
| `data/raw` e `data/processed` | Base bruta, base tratada, previsões e motivos |
| `models/` | Modelos `.joblib` e `metricas.json` |
| `reports/` | Tabelas, JSONs de monitoramento e `figuras/` |
| `docs/` | Documentação de cada etapa, arquitetura e Model Card |
| `tests/` | Testes automáticos (pytest) |

## Decisões de arquitetura

| Decisão | Motivo |
|---|---|
| App separado do treino | O app só lê arquivos, então abre rápido e não muda resultado sem querer |
| `comum.py` compartilhado | Evita regras duplicadas e inconsistentes entre etapas e app |
| Pipeline do sklearn com calibração | O pré-processamento viaja junto com o modelo e as probabilidades ficam realistas |
| Divisão temporal | Simula o uso real: treinar no passado e prever o futuro |
| MLflow só no registro e no monitoramento | O app não depende de servidor; o MLflow guarda o histórico e a governança |
| Parâmetros de negócio em um só lugar | `ParametrosNegocio` e a barra lateral do app permitem testar premissas |

## Como adaptar a dados reais
1. Troque a Etapa 1 pela leitura da base real e mantenha os nomes e tipos do dicionário de dados.
2. Ajuste `ParametrosNegocio` com o custo real e o % evitado.
3. Rode `python rodar_pipeline.py` e confira os testes com `pytest -q`.
