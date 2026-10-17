# Etapa 8 – App Streamlit

Arquivo: `app/app.py` · Rodar a partir da pasta do projeto: `streamlit run app/app.py`

> **Status de validação:** o código do app foi executado de ponta a ponta com um Streamlit simulado (inclusive o cálculo de um sinistro novo) e os dados e modelos são os reais do pipeline. Não foi aberto em navegador no ambiente de desenvolvimento. Se algo visual precisar de ajuste, é só me avisar.

## Telas

| Aba | Para quê |
|---|---|
| 📊 Visão geral | KPIs de negócio, comparação modelo x alternativas e curva "vale ampliar a equipe?" |
| 📋 Fila de investigação | Fila priorizada com filtros por analista e prioridade, motivos de alerta e download em CSV |
| 🔎 Analisar um sinistro | Formulário: probabilidade de fraude, recomendação (investigar ou liberar) e explicação |
| 🧠 Modelos e explicabilidade | Comparação dos 3 modelos, ROC/PR, calibração, ganho acumulado e importâncias |
| 📈 Monitoramento | Status, PSI, AUC e alertas (cenário real e simulado) |
| ℹ️ Sobre | Problema, solução e limites |

A barra lateral controla as premissas (custo por investigação, % evitado, capacidade e nº de analistas) e todos os números se recalculam na hora.

## O que o app precisa existir (gerado pelas etapas)

| Arquivo | Gerado por |
|---|---|
| `models/modelo_final.joblib` | Etapa 4 |
| `models/metricas.json` | Etapa 4 |
| `data/processed/predicoes_teste.csv` | Etapa 4 |
| `data/processed/motivos_teste.csv` | Etapa 5 |
| `data/processed/sinistros_tratado.csv` | Etapa 2 (valores típicos para a explicação) |
| `reports/monitoramento*.json/csv` e `reports/figuras/*.png` | Etapas 3, 4, 5 e 7 |

Se faltar algo, o app mostra qual arquivo e qual comando rodar. O atalho é `python rodar_pipeline.py`.

## Garantia de consistência
O formulário usa `construir_linha` (`src/comum.py`), que replica as variáveis da Etapa 2. O teste `test_construir_linha_replica_o_preprocessamento` compara 300 sinistros reais e falha se as duas versões divergirem.
