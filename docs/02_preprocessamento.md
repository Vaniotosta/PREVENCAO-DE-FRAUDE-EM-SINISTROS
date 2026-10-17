# Etapa 2 – Pré-processamento

Arquivo: `src/02_preprocessamento.py`
Entrada: `data/raw/sinistros_raw.csv` (20.300 linhas, 30 colunas)
Saída: `data/processed/sinistros_tratado.csv` (20.000 linhas, 45 colunas)

## Resultado em números

| Item | Quantidade |
|---|---|
| Duplicados removidos | 300 |
| Idades inválidas viradas nulo | 81 |
| Tempo de cliente inválido virado nulo | 75 |
| Rendas acima do teto (limitadas) | 79 |
| Valores segurados acima do teto (limitados) | 59 |
| Taxa de fraude | 6,76% |

## Decisões e motivos

| Decisão | Por quê |
|---|---|
| Remover `status_investigacao` e `valor_recuperado` | Só existem **depois** da investigação. Usar seria "colar" a resposta no modelo (vazamento) |
| Remover `pais` | Sempre "Brasil", não ajuda a separar nada |
| Valores impossíveis viram nulo, não são apagados | Perder a linha seria perder um sinistro real |
| Outliers limitados por **tetos de negócio** | Não usa média nem percentil da base, então não vaza informação |
| Nulos **não** são preenchidos aqui | O preenchimento é feito no `Pipeline` do sklearn (Etapa 4), calculado só com o treino |
| Colunas `*_nulo` | Não informar um dado pode ser sinal de risco |
| `id_segurado` removido | É só um identificador |

## Variáveis criadas

| Variável | Como é calculada | Por que ajuda |
|---|---|---|
| dias_inicio_ate_sinistro | data do sinistro − início da apólice | Sinistro logo após contratar é suspeito |
| apolice_nova_60d | 1 se menos de 60 dias | Versão simples da anterior |
| atraso_aviso_dias | data do aviso − data do sinistro | Aviso demorado levanta dúvida |
| aviso_tardio_10d | 1 se atraso > 10 dias | Versão simples da anterior |
| razao_sinistro_valor_segurado | valor do sinistro ÷ valor segurado | Sinistros que consomem quase todo o bem |
| razao_sinistro_premio | valor do sinistro ÷ prêmio anual | Retorno alto sobre o que foi pago |
| razao_sinistro_renda | valor do sinistro ÷ renda mensal | Valor grande para a renda |
| sinistro_madrugada | hora entre 0 e 5 | Ocorrências sem testemunha |
| sinistro_fim_semana | sábado ou domingo | Padrão de calendário |
| mes_sinistro | mês da ocorrência | Sazonalidade |
| sinistros_antes_12m | total − últimos 12 meses | Histórico mais antigo |
| frequencia_sinistros_ano | total ÷ (anos de cliente + 1) | Frequência ao longo do relacionamento |
| sem_boletim_sem_testemunha | sem B.O. e sem testemunha | Combinação de falta de provas |
| *_nulo (8 colunas) | 1 se o dado estava vazio | Falta de informação como sinal |

## Como conferir

```bash
python src/02_preprocessamento.py
```

O script imprime o relatório de qualidade no final. Conferimos também que não sobraram
duplicados, que as categorias ficaram padronizadas e que a fraude é bem mais comum em
apólices novas (13,4% contra 4,4%).

## Atualização na Etapa 4
A variável `mes_sinistro` continua sendo criada na base tratada, mas **não entra no modelo**. Como a divisão é temporal, o mês só mediria a cobertura do calendário em cada período e geraria falsos alertas de monitoramento.
