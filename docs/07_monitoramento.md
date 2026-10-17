# Etapa 7 – Monitoramento do modelo

Arquivo: `src/07_monitoramento.py`

```bash
python src/07_monitoramento.py                  # dados reais de teste
python src/07_monitoramento.py --simular-drift  # cenário de mudança (demonstração)
```

## O que é medido

| Verificação | Como | Regra |
|---|---|---|
| Mudança nas variáveis | **PSI** de cada variável: treino x dados atuais | < 0,10 estável · 0,10 a 0,25 atenção · > 0,25 alerta |
| Mudança no score | PSI das probabilidades do modelo | mesma regra |
| Queda de desempenho | AUC atual x validação | alerta se cair mais de 0,05 |
| Queda de resultado | Resultado líquido atual x validação | alerta se cair mais de 25% |

**PSI (Population Stability Index)** compara a distribuição de uma variável em dois momentos. Se ela mudou muito, o modelo pode estar vendo um mundo diferente do que aprendeu.

## Resultados

| Cenário | Status | PSI máximo | PSI do score | Variáveis em alerta | AUC |
|---|---|---|---|---|---|
| Real (teste) | **OK** | 0,006 | 0,009 | 0 | 0,831 (validação 0,824) |
| Simulado | **ALERTA** | 8,56 (`atraso_aviso_dias`) | 0,58 | 7 | 0,818 |

O cenário simulado aumenta o valor dos sinistros em 50%, atrasa os avisos em 4 dias, move 25% das ocorrências para a madrugada e reduz o score de crédito em 80 pontos. O alerta vem da **mudança nos dados**; o resultado financeiro pode até subir (sinistros mais caros), mas o modelo está operando fora do que aprendeu.

## Quando agir

| Sinal | Ação |
|---|---|
| Alerta de PSI em poucas variáveis | Investigar a origem (mudança de sistema, de produto ou de campanha) |
| PSI do score em alerta | Reavaliar os limites e a capacidade da fila |
| Queda de AUC ou de resultado | Retreinar com dados recentes (Etapas 4 e 6) |
| Rótulos novos de fraude disponíveis | Rodar o pipeline completo e comparar no MLflow |

## Arquivos gerados
`reports/monitoramento.{csv,json}`, `reports/monitoramento_simulado.{csv,json}`, `reports/figuras/monitoramento_psi*.png`.
O app mostra os dois cenários na aba "Monitoramento".

## Limite importante
Em produção, o AUC só pode ser calculado depois que os casos são investigados e a fraude é confirmada (atraso de semanas ou meses). Até lá, o PSI é o sinal mais rápido.
