"""
Módulo compartilhado do projeto.

Tudo o que mais de uma etapa (e o app) precisa fica aqui, para existir
UMA única versão de cada regra:

- caminhos e constantes
- lista de variáveis (numéricas e categóricas)
- divisão temporal treino / validação / teste
- construção dos modelos (pipeline do sklearn)
- métricas de negócio e priorização de casos
- explicação por grupos de variáveis
- PSI (monitoramento de mudança nos dados)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Caminhos e constantes
# --------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]
DADOS_BRUTOS = RAIZ / "data" / "raw" / "sinistros_raw.csv"
DADOS_TRATADOS = RAIZ / "data" / "processed" / "sinistros_tratado.csv"
PREDICOES = RAIZ / "data" / "processed" / "predicoes_teste.csv"
MOTIVOS = RAIZ / "data" / "processed" / "motivos_teste.csv"
PASTA_MODELOS = RAIZ / "models"
PASTA_REPORTS = RAIZ / "reports"
PASTA_FIGURAS = PASTA_REPORTS / "figuras"
MODELO_FINAL = PASTA_MODELOS / "modelo_final.joblib"
METRICAS_JSON = PASTA_MODELOS / "metricas.json"
MLFLOW_URI = f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIMENTO = "fraude_seguro_sinistros"
MLFLOW_MONITORAMENTO = "fraude_seguro_monitoramento"
NOME_MODELO_REGISTRADO = "modelo_fraude_seguro"

SEED = 42
ALVO = "fraude"
COL_ID = "id_sinistro"
COL_DATA = "data_sinistro"


@dataclass
class ParametrosNegocio:
    """Premissas financeiras (fictícias). Ajuste para o caso real."""

    custo_investigacao: float = 1500.0   # R$ por sinistro investigado
    taxa_prevencao: float = 0.70         # % do valor do sinistro que deixa de ser pago quando a fraude é confirmada
    capacidade_pct: float = 0.10         # % dos sinistros que a equipe consegue investigar

    def como_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Variáveis do modelo
# --------------------------------------------------------------------------
COLS_CAT = ["estado_civil", "tipo_cobertura", "canal_venda", "tipo_sinistro", "uf"]

COLS_NUM = [
    # segurado
    "idade", "tempo_cliente_anos", "score_credito", "renda_mensal",
    # apólice
    "valor_segurado", "franquia", "premio_anual", "alteracoes_apolice_90d",
    # histórico
    "qtd_sinistros_12m", "qtd_sinistros_total", "sinistros_antes_12m", "frequencia_sinistros_ano",
    # acidente
    "hora_sinistro", "boletim_ocorrencia", "qtd_testemunhas", "houve_vitima",
    "oficina_credenciada", "valor_sinistro",
    # combinações
    "dias_inicio_ate_sinistro", "apolice_nova_60d", "atraso_aviso_dias", "aviso_tardio_10d",
    "razao_sinistro_valor_segurado", "razao_sinistro_premio", "razao_sinistro_renda",
    "sinistro_madrugada", "sinistro_fim_semana", "sem_boletim_sem_testemunha",
    # indicadores de nulo
    "score_credito_nulo", "qtd_testemunhas_nulo", "oficina_credenciada_nulo",
    "renda_mensal_nulo", "estado_civil_nulo", "canal_venda_nulo", "idade_nulo",
    "tempo_cliente_anos_nulo",
]
FEATURES = COLS_NUM + COLS_CAT

# Colunas que podem ser nulas no dado real (e ganham indicador *_nulo)
COLS_COM_INDICADOR_NULO = [
    "score_credito", "qtd_testemunhas", "oficina_credenciada", "renda_mensal",
    "estado_civil", "canal_venda", "idade", "tempo_cliente_anos",
]

# Grupos usados para explicar um caso em linguagem de negócio
GRUPOS_EXPLICACAO = {
    "Sinistro logo após contratar a apólice": ["dias_inicio_ate_sinistro", "apolice_nova_60d"],
    "Atraso para avisar a seguradora": ["atraso_aviso_dias", "aviso_tardio_10d"],
    "Valor do sinistro (e peso sobre o bem)": [
        "valor_sinistro", "razao_sinistro_valor_segurado",
        "razao_sinistro_premio", "razao_sinistro_renda",
    ],
    "Boletim de ocorrência e testemunhas": [
        "boletim_ocorrencia", "qtd_testemunhas", "sem_boletim_sem_testemunha", "qtd_testemunhas_nulo",
    ],
    "Histórico de sinistros": [
        "qtd_sinistros_12m", "qtd_sinistros_total", "sinistros_antes_12m", "frequencia_sinistros_ano",
    ],
    "Alterações recentes na apólice": ["alteracoes_apolice_90d"],
    "Horário da ocorrência": ["hora_sinistro", "sinistro_madrugada"],
    "Tipo de sinistro": ["tipo_sinistro"],
    "Oficina": ["oficina_credenciada", "oficina_credenciada_nulo"],
    "Score de crédito": ["score_credito", "score_credito_nulo"],
    "Perfil do segurado": [
        "idade", "tempo_cliente_anos", "estado_civil", "renda_mensal",
        "idade_nulo", "tempo_cliente_anos_nulo", "estado_civil_nulo", "renda_mensal_nulo",
    ],
    "Dados da apólice": ["tipo_cobertura", "valor_segurado", "franquia", "premio_anual", "canal_venda", "canal_venda_nulo"],
    "Local e calendário": ["uf", "sinistro_fim_semana"],
    "Houve vítima": ["houve_vitima"],
}


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------
def carregar_tratado() -> pd.DataFrame:
    if not DADOS_TRATADOS.exists():
        raise FileNotFoundError(
            f"Não encontrei {DADOS_TRATADOS}. Rode antes: python src/02_preprocessamento.py"
        )
    return pd.read_csv(DADOS_TRATADOS, parse_dates=[COL_DATA])


def dividir_temporal(df: pd.DataFrame, pct_treino: float = 0.60, pct_validacao: float = 0.20):
    """Treino = passado, validação = meio, teste = futuro (mais realista que sorteio)."""
    df = df.sort_values([COL_DATA, COL_ID]).reset_index(drop=True)
    n = len(df)
    fim_treino = int(n * pct_treino)
    fim_valid = int(n * (pct_treino + pct_validacao))
    return df.iloc[:fim_treino].copy(), df.iloc[fim_treino:fim_valid].copy(), df.iloc[fim_valid:].copy()


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
def criar_modelos() -> dict:
    """Os 3 modelos estatísticos do projeto (todos do scikit-learn)."""
    return {
        "regressao_logistica": LogisticRegression(max_iter=2000, C=0.5),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=10, max_features="sqrt", n_jobs=-1, random_state=SEED
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=250, random_state=SEED
        ),
    }


def construir_pipeline(estimador) -> Pipeline:
    """Imputação + escala + one-hot + modelo calibrado.

    A calibração é essencial: o valor esperado de cada caso usa a probabilidade
    de fraude, então ela precisa ser uma probabilidade "de verdade".
    """
    numericas = Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())])
    categoricas = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([("num", numericas, COLS_NUM), ("cat", categoricas, COLS_CAT)])
    return Pipeline([
        ("preprocessamento", pre),
        ("modelo", CalibratedClassifierCV(estimador, method="sigmoid", cv=3)),
    ])


# --------------------------------------------------------------------------
# Métricas de negócio
# --------------------------------------------------------------------------
def valor_esperado_liquido(prob, valor_sinistro, p: ParametrosNegocio):
    """Quanto esperamos proteger investigando o caso, já descontando o custo."""
    return np.asarray(prob) * np.asarray(valor_sinistro) * p.taxa_prevencao - p.custo_investigacao


def selecionar_casos(prob, valor_sinistro, p: ParametrosNegocio, capacidade_pct: float | None = None) -> np.ndarray:
    """Escolhe os casos a investigar: maiores valores esperados, dentro da capacidade e só se > 0."""
    score = valor_esperado_liquido(prob, valor_sinistro, p)
    cap = p.capacidade_pct if capacidade_pct is None else capacidade_pct
    k = int(round(len(score) * cap))
    mascara = np.zeros(len(score), dtype=bool)
    if k <= 0:
        return mascara
    topo = np.argsort(-score, kind="stable")[:k]
    mascara[topo[score[topo] > 0]] = True
    return mascara


def metricas_negocio(y, prob, valor_sinistro, p: ParametrosNegocio, capacidade_pct: float | None = None) -> dict:
    y = np.asarray(y)
    prob = np.asarray(prob, dtype=float)
    valor = np.asarray(valor_sinistro, dtype=float)
    mascara = selecionar_casos(prob, valor, p, capacidade_pct)
    k = int(mascara.sum())
    fraudes_total = int(y.sum())
    capturadas = int(y[mascara].sum())
    protegido = float((y * valor * p.taxa_prevencao)[mascara].sum())
    custo = k * p.custo_investigacao
    taxa_base = y.mean()
    precisao = capturadas / k if k else 0.0
    return {
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "casos_investigados": k,
        "fraudes_total": fraudes_total,
        "fraudes_capturadas": capturadas,
        "recall_na_capacidade": capturadas / fraudes_total if fraudes_total else 0.0,
        "precisao_na_capacidade": precisao,
        "lift_na_capacidade": precisao / taxa_base if taxa_base else 0.0,
        "valor_protegido": protegido,
        "custo_investigacao": float(custo),
        "resultado_liquido": protegido - custo,
        "roi": (protegido - custo) / custo if custo else 0.0,
    }


def comparar_estrategias(y, prob, valor_sinistro, p: ParametrosNegocio, capacidade_pct: float | None = None) -> pd.DataFrame:
    """Compara o modelo com alternativas simples (sem modelo)."""
    y = np.asarray(y)
    valor = np.asarray(valor_sinistro, dtype=float)
    n = len(y)
    cap = p.capacidade_pct if capacidade_pct is None else capacidade_pct
    k = int(round(n * cap))
    ganho_real = y * valor * p.taxa_prevencao

    def linha(nome, mascara):
        return {
            "estrategia": nome,
            "casos_investigados": int(mascara.sum()),
            "fraudes_capturadas": int(y[mascara].sum()),
            "valor_protegido": float(ganho_real[mascara].sum()),
            "custo": float(mascara.sum() * p.custo_investigacao),
        }

    def topo(score):
        m = np.zeros(n, dtype=bool)
        m[np.argsort(-score, kind="stable")[:k]] = True
        return m

    linhas = [
        linha("Não investigar", np.zeros(n, dtype=bool)),
        linha("Investigar todos", np.ones(n, dtype=bool)),
        linha("Maiores valores (sem modelo)", topo(valor)),
        linha("Modelo + valor esperado", selecionar_casos(prob, valor, p, cap)),
        linha("Ideal (só fraudes, impossível)", topo(ganho_real)),
    ]
    # Aleatório: valor esperado de sortear k casos
    protegido_alea = k * float(ganho_real.mean())
    linhas.insert(2, {
        "estrategia": "Aleatório (média)", "casos_investigados": k,
        "fraudes_capturadas": int(round(k * float(y.mean()))),
        "valor_protegido": protegido_alea, "custo": float(k * p.custo_investigacao),
    })
    tabela = pd.DataFrame(linhas)
    tabela["resultado_liquido"] = tabela["valor_protegido"] - tabela["custo"]
    return tabela


def curva_capacidade(y, prob, valor_sinistro, p: ParametrosNegocio, passos=None) -> pd.DataFrame:
    """Resultado líquido para diferentes capacidades de investigação."""
    passos = passos if passos is not None else np.round(np.arange(0.01, 0.41, 0.01), 2)
    linhas = []
    for cap in passos:
        m = metricas_negocio(y, prob, valor_sinistro, p, capacidade_pct=float(cap))
        linhas.append({"capacidade_pct": float(cap), **{k: m[k] for k in (
            "casos_investigados", "fraudes_capturadas", "valor_protegido",
            "custo_investigacao", "resultado_liquido", "roi")}})
    return pd.DataFrame(linhas)


def distribuir_analistas(ordem_prioridade: pd.Series, n_analistas: int) -> list[str]:
    """Distribui casos em 'serpentina' (1,2,3,3,2,1,...) para equilibrar a carga e a dificuldade."""
    n_analistas = max(1, int(n_analistas))
    nomes = []
    for i in range(len(ordem_prioridade)):
        rodada, posicao = divmod(i, n_analistas)
        idx = posicao if rodada % 2 == 0 else n_analistas - 1 - posicao
        nomes.append(f"Analista {idx + 1}")
    return nomes


def rotular_prioridade(posicoes: pd.Series, total: int) -> pd.Series:
    """Crítica = primeiros 25% da fila, Alta = próximos 25%, Média = o restante."""
    rel = posicoes / max(total, 1)
    return pd.Series(np.where(rel <= 0.25, "Crítica", np.where(rel <= 0.50, "Alta", "Média")), index=posicoes.index)


# --------------------------------------------------------------------------
# Explicação por grupos (funciona com qualquer modelo do sklearn)
# --------------------------------------------------------------------------
def valores_tipicos(df_treino: pd.DataFrame) -> dict:
    """Valor 'normal' de cada variável: mediana (números) ou moda (categorias)."""
    tipicos = {}
    for col in COLS_NUM:
        tipicos[col] = float(df_treino[col].median())
    for col in COLS_CAT:
        tipicos[col] = df_treino[col].mode().iloc[0]
    return tipicos


def explicar_lote(modelo, X: pd.DataFrame, tipicos: dict) -> pd.DataFrame:
    """Contribuição de cada grupo para a probabilidade de fraude.

    Troca o grupo pelos valores típicos e mede quanto a probabilidade cai.
    Positivo = o grupo AUMENTA o risco deste caso.
    """
    X = X[FEATURES].reset_index(drop=True)
    base = modelo.predict_proba(X)[:, 1]
    contribuicoes = {}
    for grupo, colunas in GRUPOS_EXPLICACAO.items():
        alterado = X.copy()
        for col in colunas:
            alterado[col] = tipicos[col]
        contribuicoes[grupo] = base - modelo.predict_proba(alterado)[:, 1]
    return pd.DataFrame(contribuicoes)


def principais_motivos(contribuicoes: pd.DataFrame, minimo: float = 0.005) -> pd.DataFrame:
    """Dois principais motivos de alerta (só grupos que aumentam o risco)."""
    motivos = []
    for _, linha in contribuicoes.iterrows():
        positivos = linha[linha > minimo].sort_values(ascending=False)
        nomes = list(positivos.index[:2]) + ["-", "-"]
        motivos.append((nomes[0], nomes[1]))
    return pd.DataFrame(motivos, columns=["motivo_principal", "motivo_secundario"])


# --------------------------------------------------------------------------
# Monitoramento: PSI (Population Stability Index)
# --------------------------------------------------------------------------
def calcular_psi(referencia: pd.Series, atual: pd.Series, faixas: int = 10) -> float:
    """PSI < 0,10 estável | 0,10 a 0,25 atenção | > 0,25 mudança forte."""
    ref = referencia.dropna()
    atu = atual.dropna()
    if ref.empty or atu.empty:
        return float("nan")
    if ref.nunique() > faixas and pd.api.types.is_numeric_dtype(ref):
        limites = np.unique(np.quantile(ref, np.linspace(0, 1, faixas + 1)))
        limites[0], limites[-1] = -np.inf, np.inf
        p_ref = np.histogram(ref, bins=limites)[0] / len(ref)
        p_atu = np.histogram(atu, bins=limites)[0] / len(atu)
    else:
        categorias = sorted(set(ref.unique()) | set(atu.unique()), key=str)
        p_ref = np.array([(ref == c).mean() for c in categorias])
        p_atu = np.array([(atu == c).mean() for c in categorias])
    p_ref = np.clip(p_ref, 1e-4, None)
    p_atu = np.clip(p_atu, 1e-4, None)
    return float(np.sum((p_atu - p_ref) * np.log(p_atu / p_ref)))


def status_psi(valor: float) -> str:
    if np.isnan(valor):
        return "sem dados"
    if valor < 0.10:
        return "estável"
    if valor < 0.25:
        return "atenção"
    return "alerta"


# --------------------------------------------------------------------------
# Linha única (usada pelo app): replica as variáveis criadas na Etapa 2
# --------------------------------------------------------------------------
def construir_linha(entrada: dict) -> pd.DataFrame:
    """Transforma os campos digitados no app em uma linha com todas as variáveis do modelo.

    Valores None são tratados como "não informado" (viram nulo + indicador *_nulo).
    """
    d = {k: (np.nan if v is None else v) for k, v in entrada.items()}
    for col in COLS_COM_INDICADOR_NULO:
        d[f"{col}_nulo"] = int(pd.isna(d.get(col, np.nan)))

    def num(chave):
        return float(d[chave]) if not pd.isna(d[chave]) else np.nan

    dias = num("dias_inicio_ate_sinistro")
    atraso = num("atraso_aviso_dias")
    valor = num("valor_sinistro")
    d["apolice_nova_60d"] = int(dias < 60)
    d["aviso_tardio_10d"] = int(atraso > 10)
    d["razao_sinistro_valor_segurado"] = valor / num("valor_segurado")
    d["razao_sinistro_premio"] = valor / num("premio_anual")
    d["razao_sinistro_renda"] = valor / num("renda_mensal") if not pd.isna(d["renda_mensal"]) else np.nan
    d["sinistro_madrugada"] = int(0 <= d["hora_sinistro"] <= 5)
    d["sinistro_fim_semana"] = int(d["dia_semana"] >= 5)
    d["sinistros_antes_12m"] = max(d["qtd_sinistros_total"] - d["qtd_sinistros_12m"], 0)
    tempo = 0.0 if pd.isna(d["tempo_cliente_anos"]) else float(d["tempo_cliente_anos"])
    d["frequencia_sinistros_ano"] = d["qtd_sinistros_total"] / (tempo + 1)
    testemunhas = 0 if pd.isna(d["qtd_testemunhas"]) else d["qtd_testemunhas"]
    d["sem_boletim_sem_testemunha"] = int(d["boletim_ocorrencia"] == 0 and testemunhas == 0)
    return pd.DataFrame([{col: d[col] for col in FEATURES}])
