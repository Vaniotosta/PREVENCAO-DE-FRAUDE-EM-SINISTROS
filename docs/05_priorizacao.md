# Etapa 5 – Priorização da fila de investigação

Arquivo: `src/05_priorizacao.py` · Rodar: `python src/05_priorizacao.py --capacidade 0.10 --analistas 5`

## Regra de priorização

```
valor esperado líquido = prob. de fraude × valor do sinistro × % evitado − custo da investigação
```

1. Calcula o valor esperado líquido de cada sinistro.
2. Ordena do maior para o menor.
3. Pega só os casos com valor **positivo**, até o limite da **capacidade** (10% por padrão).
4. Classifica a prioridade: **Crítica** (primeiros 25%), **Alta** (próximos 25%) e **Média** (restante).
5. Divide entre os analistas em **serpentina** (1-2-3-3-2-1…), que equilibra carga e dificuldade.

Um sinistro de valor alto com probabilidade média pode vir antes de um de valor baixo com probabilidade alta, porque o que importa é o dinheiro protegido.

## Resultado (base de teste, 4.000 sinistros)

| Estratégia | Casos | Fraudes | Resultado líquido |
|---|---|---|---|
| Não investigar | 0 | 0 | R$ 0 |
| Investigar todos | 4.000 | 303 | −R$ 0,48 mi |
| Aleatório (média) | 400 | ~30 | −R$ 0,05 mi |
| Maiores valores (sem modelo) | 400 | 58 | R$ 1,54 mi |
| **Modelo + valor esperado** | **400** | **141** | **R$ 2,90 mi** |
| Ideal (impossível) | 400 | 303 | R$ 4,92 mi |

- A fila de 400 casos encontra 141 das 303 fraudes (35,2% de precisão contra 7,6% da base).
- A fila tem 100 casos Críticos, 100 Altos e 200 Médios, com 80 casos por analista.
- Em 202 dos 400 casos o motivo principal é "sinistro logo após contratar a apólice".
- **Capacidade ideal:** na curva, o melhor resultado aparece com 20% de investigação (R$ 3,24 mi, ROI 2,7x). A partir daí cada caso extra custa mais do que protege.

## Arquivos gerados
`reports/fila_priorizada.csv`, `reports/comparacao_estrategias.csv`, `data/processed/motivos_teste.csv`, `reports/figuras/prioridade_01_estrategias.png`, `prioridade_02_curva_capacidade.png`.

## Premissas (ajuste ao caso real)
Custo por investigação, % evitado e capacidade ficam em `ParametrosNegocio` (`src/comum.py`) e nos controles da barra lateral do app.
