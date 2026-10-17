# Etapa 3 – Análise Exploratória (EDA)

Arquivo: `src/03_eda.py` · Entrada: `data/processed/sinistros_tratado.csv` · Saída: `reports/figuras/eda_*.png` e `reports/eda_resumo.json`

## Pergunta que a EDA responde
Onde a fraude aparece mais e quais sinais separam um sinistro fraudulento de um legítimo?

## Principais descobertas (base de 20.000 sinistros, 6,8% de fraude)

| Sinal | Taxa de fraude | Comparação |
|---|---|---|
| Aviso com mais de 10 dias de atraso | 17,4% | média geral 6,8% |
| Sinistro maior que 45% do valor segurado | 15,6% | 3,5% nos demais |
| Apólice com menos de 60 dias | 13,4% | 4,4% nas mais antigas |
| Sem boletim de ocorrência | 12,6% | 5,5% com B.O. |
| Ocorrência de madrugada (0h–5h) | 9,6% | média geral 6,8% |
| Oficina não credenciada | 9,3% | 5,6% credenciada |
| Incêndio, furto e roubo | 9,7% a 10,3% | 5,3% a 5,6% em colisão e danos naturais |

- Sinistros fraudulentos têm valor médio maior (R$ 25,6 mil contra R$ 19,6 mil).
- Variáveis mais correlacionadas com fraude: razão sinistro/valor segurado (0,16), apólice nova (0,16), razão sinistro/prêmio (0,14), B.O. (−0,11) e score de crédito (−0,10).
- Estado e canal de venda quase não mudam a taxa (variação de 6% a 8%).

## Gráficos gerados (`reports/figuras/`)

| Arquivo | O que mostra |
|---|---|
| eda_01_desbalanceamento.png | Poucos casos de fraude: exige métricas adequadas |
| eda_02_taxa_por_categoria.png | Taxa de fraude por tipo, cobertura, canal, B.O., oficina e estado |
| eda_03_tempo_apolice.png | Fraude x tempo entre contratar e sinistrar |
| eda_04_razao_valor.png | Distribuição do valor do sinistro ÷ valor segurado |
| eda_05_hora.png | Fraude por hora do dia |
| eda_06_aviso_historico.png | Atraso no aviso e sinistros nos últimos 12 meses |
| eda_07_valor_boxplot.png | Valor do sinistro: fraude x legítimo |
| eda_08_correlacao.png | Variáveis mais ligadas à fraude |
| eda_09_nulos.png | Nulos que seguem para o Pipeline |

## O que isso muda na modelagem
1. A classe rara pede métricas como PR-AUC, recall e precisão na capacidade, e não só acurácia.
2. As variáveis de combinação criadas na Etapa 2 (razões, tempo de apólice, atraso) são as mais úteis.
3. Estado e canal de venda têm pouco sinal, mas ficam no modelo para o Pipeline decidir.
