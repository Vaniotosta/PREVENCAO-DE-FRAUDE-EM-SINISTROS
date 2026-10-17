# Dicionário de Dados – `sinistros_raw.csv`

Base **fictícia** de sinistros de seguro auto. Cada linha é um sinistro avisado à seguradora.
Alvo do modelo: `fraude` (1 = fraude confirmada, 0 = sinistro legítimo).

## Segurado

| Coluna | Tipo esperado | Descrição |
|---|---|---|
| id_segurado | texto | Código do segurado (fictício) |
| idade | inteiro | Idade em anos (18–85 esperado) |
| tempo_cliente_anos | decimal | Anos de relacionamento com a seguradora |
| score_credito | inteiro | Score de crédito (250–950) |
| estado_civil | categoria | Solteiro, Casado, Divorciado, Viúvo |
| renda_mensal | decimal | Renda mensal em R$ |

## Apólice

| Coluna | Tipo esperado | Descrição |
|---|---|---|
| tipo_cobertura | categoria | Compreensiva, Roubo e Furto, Colisão, Básica |
| valor_segurado | decimal | Valor do bem segurado (R$) |
| franquia | decimal | Valor da franquia (R$) |
| premio_anual | decimal | Prêmio anual pago (R$) |
| canal_venda | categoria | Corretor, Online, Agência, Telefone |
| alteracoes_apolice_90d | inteiro | Alterações na apólice nos 90 dias antes do sinistro |
| data_inicio_apolice | data | Início da vigência |

## Acidente / Sinistro

| Coluna | Tipo esperado | Descrição |
|---|---|---|
| id_sinistro | texto | Código do sinistro |
| data_sinistro | data | Data da ocorrência |
| data_aviso | data | Data em que o segurado avisou a seguradora |
| hora_sinistro | inteiro | Hora do dia (0–23) |
| tipo_sinistro | categoria | Colisão, Roubo, Furto, Incêndio, Danos naturais |
| uf | categoria | Estado da ocorrência |
| boletim_ocorrencia | categoria | Se existe B.O. (Sim/Não) |
| qtd_testemunhas | inteiro | Número de testemunhas |
| houve_vitima | categoria | Houve vítima? (Sim/Não) |
| oficina_credenciada | categoria | Reparo em oficina credenciada? (Sim/Não) |
| valor_sinistro | decimal | Valor estimado do sinistro (R$) |

## Histórico

| Coluna | Tipo esperado | Descrição |
|---|---|---|
| qtd_sinistros_12m | inteiro | Sinistros do segurado nos últimos 12 meses |
| qtd_sinistros_total | inteiro | Sinistros do segurado em todo o histórico |

## Alvo e colunas proibidas

| Coluna | Uso | Descrição |
|---|---|---|
| **fraude** | **ALVO** | 1 = fraude, 0 = legítimo |
| status_investigacao | ⛔ vazamento | Só existe depois da investigação. **Não usar como variável** |
| valor_recuperado | ⛔ vazamento | Só existe depois da investigação. **Não usar como variável** |
| pais | ⛔ sem informação | Sempre "Brasil" (coluna constante) |

## Problemas de qualidade plantados (para tratar no pandas)

| # | Problema | Onde aparece | Tratamento previsto (Etapa 2) |
|---|---|---|---|
| 1 | Linhas duplicadas (~1,5%) | base inteira | `drop_duplicates` |
| 2 | Valores nulos | testemunhas, renda, score, oficina, estado civil, canal | Imputação por mediana/moda + indicador de nulo |
| 3 | Valores impossíveis | `idade` (-1, 0, 150, 999), `tempo_cliente_anos` = -1 | Marcar como nulo e imputar |
| 4 | Dinheiro como texto | `valor_sinistro` ("R$ 12.345,67") | Converter para número |
| 5 | Datas em dois formatos | `AAAA-MM-DD` e `DD/MM/AAAA` | Converter com `pd.to_datetime` |
| 6 | Categorias inconsistentes | caixa, espaços, acentos, "Bahia" x "BA", "S" x "Sim" | Padronizar texto |
| 7 | Outliers extremos | `renda_mensal`, `valor_segurado` | Limitar (winsorização) |
| 8 | Vazamento de dados | `status_investigacao`, `valor_recuperado` | Remover antes de treinar |
| 9 | Coluna constante | `pais` | Remover |
| 10 | Classe desbalanceada | `fraude` (~7%) | `class_weight` e métricas adequadas |
