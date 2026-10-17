# Model Card – Detecção de sinistros suspeitos

## Resumo
- **Modelo em uso:** Regressão Logística calibrada (scikit-learn), escolhida por maior resultado líquido na validação.
- **Tarefa:** estimar a probabilidade de fraude de cada sinistro de seguro auto e priorizar investigações.
- **Versão dos dados:** base sintética de 20.000 sinistros (2023 a 2025), 6,8% de fraude.

## Uso pretendido
Apoiar analistas na ordem de investigação. A decisão final e a confirmação da fraude são sempre humanas. **Não** deve negar sinistros de forma automática.

## Dados
Segurado, apólice, acidente e histórico de sinistros. Dados **100% fictícios**. Colunas pós-investigação (`status_investigacao`, `valor_recuperado`) foram removidas para evitar vazamento.

## Desempenho (teste: sinistros de jun/2025 a dez/2025)

| Métrica | Valor |
|---|---|
| ROC-AUC | 0,831 |
| PR-AUC | 0,373 (taxa base 7,6%) |
| Recall com 10% de capacidade | 46,5% |
| Precisão com 10% de capacidade | 35,2% (4,65x melhor que sortear) |
| Resultado líquido simulado | R$ 2,90 mi (valor protegido R$ 3,50 mi − custo R$ 0,60 mi) |

Os outros dois modelos testados tiveram desempenho próximo (AUC 0,827 e 0,837).

## Limitações
1. **Dados sintéticos:** os padrões foram criados por nós. O desempenho com dados reais será diferente e menor.
2. **Premissas financeiras ilustrativas:** custo de R$ 1.500 e 70% de valor evitado precisam ser substituídos pelos valores reais.
3. **Viés de seleção em dados reais:** só sabemos quem é fraudador entre os casos investigados. Fraudes nunca investigadas aparecem como "legítimas" e o modelo aprende o critério antigo da equipe.
4. **Mudança ao longo do tempo:** fraudadores adaptam o comportamento. Monitore o PSI e retreine (Etapa 7).
5. **Probabilidades dependem da taxa de fraude:** se ela mudar muito, recalibre o modelo.

## Cuidados éticos e de justiça
- O modelo usa idade, estado civil e estado (UF). Em produção, avalie com jurídico e compliance se essas variáveis são aceitáveis e meça se a fila afeta grupos de forma desigual.
- Mantenha o motivo de cada alerta (explicação por grupos) disponível ao analista.
- Defina um processo para o segurado contestar uma investigação.

## Manutenção
Reavalie a cada ciclo de novos rótulos de fraude. Compare as versões no MLflow antes de trocar o `champion`.
