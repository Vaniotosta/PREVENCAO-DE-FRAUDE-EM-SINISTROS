"""
ETAPA 1 - Geração do dataset sintético de sinistros de seguro auto.

Objetivo
--------
Criar uma base realista (e propositalmente "suja") para treinar um modelo de
detecção de fraude. Os dados são 100% fictícios: nenhum CPF, nome ou dado real.

Como rodar
----------
    python src/01_gerar_dados.py

Saída
-----
    data/raw/sinistros_raw.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_SINISTROS = 20_000
RAIZ = Path(__file__).resolve().parents[1]
ARQUIVO_SAIDA = RAIZ / "data" / "raw" / "sinistros_raw.csv"


# --------------------------------------------------------------------------
# 1) Base "limpa": variáveis de negócio + regra de fraude escondida
# --------------------------------------------------------------------------
def gerar_base_limpa(n: int, rng: np.random.Generator) -> pd.DataFrame:
    df = pd.DataFrame({"id_sinistro": [f"SIN-{i:06d}" for i in range(1, n + 1)]})
    df["id_segurado"] = [f"SEG-{i:05d}" for i in rng.integers(1, int(n * 0.75), n)]

    # ---- Segurado ----
    df["idade"] = np.clip(rng.normal(42, 13, n), 18, 85).round().astype(int)
    df["tempo_cliente_anos"] = np.clip(rng.exponential(4, n), 0, 30).round(1)
    df["score_credito"] = np.clip(rng.normal(650, 120, n), 250, 950).round().astype(int)
    df["estado_civil"] = rng.choice(
        ["Solteiro", "Casado", "Divorciado", "Viúvo"], n, p=[0.35, 0.45, 0.15, 0.05]
    )
    df["renda_mensal"] = np.clip(rng.lognormal(8.5, 0.6, n), 1500, 60000).round(2)

    # ---- Apólice ----
    df["tipo_cobertura"] = rng.choice(
        ["Compreensiva", "Roubo e Furto", "Colisão", "Básica"], n, p=[0.45, 0.2, 0.2, 0.15]
    )
    df["valor_segurado"] = np.clip(rng.lognormal(10.9, 0.5, n), 15000, 400000).round(2)
    df["franquia"] = (df["valor_segurado"] * rng.uniform(0.03, 0.10, n)).round(2)
    df["premio_anual"] = (df["valor_segurado"] * rng.uniform(0.03, 0.07, n)).round(2)
    df["canal_venda"] = rng.choice(
        ["Corretor", "Online", "Agência", "Telefone"], n, p=[0.5, 0.25, 0.15, 0.1]
    )
    df["alteracoes_apolice_90d"] = rng.poisson(0.4, n)

    # ---- Histórico ----
    df["qtd_sinistros_12m"] = rng.poisson(0.35, n)
    df["qtd_sinistros_total"] = df["qtd_sinistros_12m"] + rng.poisson(0.8, n)

    # ---- Datas ----
    # Sinistros distribuídos de forma estável entre 2023 e 2025 (sem "tendência" artificial);
    # o início da apólice é calculado para trás a partir da data do sinistro.
    dias_desde_inicio = np.clip(rng.exponential(200, n), 1, 1000).astype(int)
    sinistro = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 1095, n), unit="D")
    df["data_sinistro"] = sinistro
    df["data_inicio_apolice"] = sinistro - pd.to_timedelta(dias_desde_inicio, unit="D")
    atraso_aviso = np.clip(rng.exponential(2, n), 0, 60).astype(int)
    df["data_aviso"] = df["data_sinistro"] + pd.to_timedelta(atraso_aviso, unit="D")

    # ---- Acidente ----
    df["tipo_sinistro"] = rng.choice(
        ["Colisão", "Roubo", "Furto", "Incêndio", "Danos naturais"],
        n,
        p=[0.55, 0.15, 0.12, 0.05, 0.13],
    )
    df["hora_sinistro"] = rng.choice(range(24), n, p=_pesos_horas())
    df["uf"] = rng.choice(
        ["SP", "RJ", "MG", "BA", "PR", "RS", "PE", "SC"], n,
        p=[0.30, 0.15, 0.12, 0.10, 0.09, 0.08, 0.08, 0.08],
    )
    df["boletim_ocorrencia"] = rng.choice(["Sim", "Não"], n, p=[0.82, 0.18])
    df["qtd_testemunhas"] = rng.poisson(1.2, n)
    df["houve_vitima"] = rng.choice(["Sim", "Não"], n, p=[0.12, 0.88])
    df["oficina_credenciada"] = rng.choice(["Sim", "Não"], n, p=[0.7, 0.3])
    fator = rng.uniform(0.05, 0.6, n)
    df["valor_sinistro"] = (df["valor_segurado"] * fator).round(2)

    # ---- Regra escondida de fraude (o modelo precisa descobrir isso) ----
    dias = (df["data_sinistro"] - df["data_inicio_apolice"]).dt.days
    razao = df["valor_sinistro"] / df["valor_segurado"]
    noite = df["hora_sinistro"].isin([0, 1, 2, 3, 4, 5])
    z = (
        -5.7
        + 1.4 * (dias < 60)
        + 1.1 * (df["boletim_ocorrencia"] == "Não")
        + 0.7 * (df["qtd_testemunhas"] == 0)
        + 0.9 * (atraso_aviso > 10)
        + 0.6 * df["qtd_sinistros_12m"].clip(upper=4)
        + 0.5 * df["alteracoes_apolice_90d"].clip(upper=3)
        + 2.0 * (razao > 0.45)
        + 0.6 * (df["oficina_credenciada"] == "Não")
        + 0.5 * noite
        + 0.8 * df["tipo_sinistro"].isin(["Roubo", "Furto", "Incêndio"])
        - 0.004 * (df["score_credito"] - 650)
    )
    prob = 1 / (1 + np.exp(-z))
    df["fraude"] = (rng.random(n) < prob).astype(int)
    return df


def _pesos_horas() -> np.ndarray:
    pesos = np.array([1, 1, 1, 1, 1, 2, 3, 5, 6, 5, 4, 4, 5, 4, 4, 4, 5, 7, 8, 6, 4, 3, 2, 1], float)
    return pesos / pesos.sum()


# --------------------------------------------------------------------------
# 2) Sujeira proposital: o que vamos tratar no pandas nas próximas etapas
# --------------------------------------------------------------------------
def sujar_base(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    # Vazamento de dados (data leakage): colunas que só existem DEPOIS da investigação
    df["status_investigacao"] = np.where(
        df["fraude"] == 1,
        rng.choice(["Fraude confirmada", "Em análise"], n, p=[0.9, 0.1]),
        rng.choice(["Arquivado", "Em análise"], n, p=[0.85, 0.15]),
    )
    df["valor_recuperado"] = np.where(df["fraude"] == 1, df["valor_sinistro"] * 0.6, 0.0).round(2)

    # Coluna constante (sem informação)
    df["pais"] = "Brasil"

    # Valores monetários como texto: "R$ 12.345,67"
    df["valor_sinistro"] = df["valor_sinistro"].astype(object)
    idx = rng.random(n) < 0.30
    df.loc[idx, "valor_sinistro"] = df.loc[idx, "valor_sinistro"].apply(_formata_brl)

    # Datas em formatos misturados
    for col in ["data_inicio_apolice", "data_sinistro", "data_aviso"]:
        df[col] = df[col].dt.strftime("%Y-%m-%d").astype(object)
        idx = rng.random(n) < 0.20
        df.loc[idx, col] = pd.to_datetime(df.loc[idx, col]).dt.strftime("%d/%m/%Y")

    # Categorias inconsistentes (caixa, espaços, sinônimos)
    df["tipo_cobertura"] = df["tipo_cobertura"].apply(_bagunca_texto, rng=rng)
    df["tipo_sinistro"] = df["tipo_sinistro"].apply(_bagunca_texto, rng=rng)
    df["uf"] = df["uf"].apply(
        lambda v: rng.choice([v, v.lower(), {"BA": "Bahia", "SP": "São Paulo"}.get(v, v)],
                             p=[0.85, 0.1, 0.05])
    )
    df["boletim_ocorrencia"] = df["boletim_ocorrencia"].apply(
        lambda v: rng.choice([v, v.upper(), "S" if v == "Sim" else "N"], p=[0.88, 0.07, 0.05])
    )

    # Valores nulos (em proporções diferentes por coluna)
    for col, pct in {
        "score_credito": 0.08, "qtd_testemunhas": 0.12, "oficina_credenciada": 0.06,
        "renda_mensal": 0.10, "estado_civil": 0.03, "canal_venda": 0.02,
    }.items():
        df.loc[rng.random(n) < pct, col] = np.nan

    # Valores impossíveis
    idx = rng.random(n) < 0.004
    df.loc[idx, "idade"] = rng.choice([-1, 0, 150, 999], idx.sum())
    idx = rng.random(n) < 0.003
    df.loc[idx, "tempo_cliente_anos"] = -1

    # Outliers extremos em valores
    idx = rng.random(n) < 0.005
    df.loc[idx, "renda_mensal"] = df.loc[idx, "renda_mensal"] * 20
    df.loc[idx, "valor_segurado"] = df.loc[idx, "valor_segurado"] * 8

    # Linhas duplicadas (1,5%)
    dup = df.sample(frac=0.015, random_state=SEED)
    df = pd.concat([df, dup], ignore_index=True)

    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def _formata_brl(v: float) -> str:
    texto = f"{float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {texto}"


def _bagunca_texto(valor: str, rng: np.random.Generator) -> str:
    return rng.choice(
        [valor, valor.upper(), valor.lower(), f" {valor} ", valor.replace("ã", "a").replace("ç", "c")],
        p=[0.7, 0.1, 0.1, 0.05, 0.05],
    )


# --------------------------------------------------------------------------
# 3) Execução
# --------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(SEED)
    limpa = gerar_base_limpa(N_SINISTROS, rng)
    suja = sujar_base(limpa, rng)

    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    suja.to_csv(ARQUIVO_SAIDA, index=False)

    print(f"Arquivo salvo em: {ARQUIVO_SAIDA}")
    print(f"Linhas: {len(suja):,} | Colunas: {suja.shape[1]}")
    print(f"Taxa de fraude (base limpa): {limpa['fraude'].mean():.2%}")
    print(f"Duplicados: {suja.duplicated().sum()}")
    print("Nulos por coluna (top 6):")
    print(suja.isna().sum().sort_values(ascending=False).head(6).to_string())


if __name__ == "__main__":
    main()
