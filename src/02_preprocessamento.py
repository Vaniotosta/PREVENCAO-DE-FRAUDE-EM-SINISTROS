"""
ETAPA 2 - Pré-processamento (limpeza + variáveis novas) com pandas.

O que este arquivo faz
----------------------
1. Remove duplicados, colunas com vazamento e coluna constante
2. Padroniza textos e categorias
3. Converte dinheiro (texto) e datas (formatos misturados)
4. Trata valores impossíveis e outliers com regras de negócio
5. Cria variáveis novas (segurado + apólice + acidente + histórico)
6. Marca onde havia valores nulos (a imputação é feita no Pipeline do sklearn,
   na Etapa 4, para NÃO vazar informação do teste para o treino)

Como rodar
----------
    python src/02_preprocessamento.py

Entrada : data/raw/sinistros_raw.csv
Saída   : data/processed/sinistros_tratado.csv
"""
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA = RAIZ / "data" / "raw" / "sinistros_raw.csv"
SAIDA = RAIZ / "data" / "processed" / "sinistros_tratado.csv"

COLUNAS_VAZAMENTO = ["status_investigacao", "valor_recuperado"]
COLUNAS_CONSTANTES = ["pais"]
COLUNAS_COM_NULOS = [
    "score_credito", "qtd_testemunhas", "oficina_credenciada",
    "renda_mensal", "estado_civil", "canal_venda",
]

# Limites de negócio (regras da seguradora, não calculados com os dados)
LIMITES = {
    "idade": (18, 85),
    "tempo_cliente_anos": (0, 30),
    "score_credito": (250, 950),
    "renda_mensal": (1_500, 60_000),
    "valor_segurado": (15_000, 400_000),
}


# --------------------------------------------------------------------------
# Funções auxiliares
# --------------------------------------------------------------------------
def tirar_acentos(texto):
    if pd.isna(texto):
        return texto
    base = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in base if not unicodedata.combining(c))


def padronizar_texto(serie: pd.Series) -> pd.Series:
    """minúsculo, sem acento, sem espaços nas pontas."""
    return serie.apply(tirar_acentos).str.strip().str.lower()


def converter_dinheiro(valor):
    """'R$ 12.345,67' -> 12345.67 | '64031.04' -> 64031.04"""
    if pd.isna(valor):
        return np.nan
    texto = str(valor).strip()
    if "R$" in texto:
        texto = texto.replace("R$", "").strip().replace(".", "").replace(",", ".")
    return float(texto)


def converter_data(serie: pd.Series) -> pd.Series:
    """Aceita 'AAAA-MM-DD' e 'DD/MM/AAAA'."""
    iso = pd.to_datetime(serie, format="%Y-%m-%d", errors="coerce")
    br = pd.to_datetime(serie, format="%d/%m/%Y", errors="coerce")
    return iso.fillna(br)


# --------------------------------------------------------------------------
# Passos do pré-processamento
# --------------------------------------------------------------------------
def remover_colunas_e_duplicados(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    log["linhas_iniciais"] = len(df)
    df = df.drop_duplicates().copy()
    log["duplicados_removidos"] = log["linhas_iniciais"] - len(df)
    df = df.drop(columns=COLUNAS_VAZAMENTO + COLUNAS_CONSTANTES)
    return df


def padronizar_categorias(df: pd.DataFrame) -> pd.DataFrame:
    # Sim/Não -> 1/0 (aceita "sim", "s", "nao", "n")
    mapa_binario = {"sim": 1, "s": 1, "nao": 0, "n": 0}
    for col in ["boletim_ocorrencia", "houve_vitima", "oficina_credenciada"]:
        df[col] = padronizar_texto(df[col]).map(mapa_binario)

    mapa_cobertura = {
        "compreensiva": "Compreensiva", "roubo e furto": "Roubo e Furto",
        "colisao": "Colisão", "basica": "Básica",
    }
    df["tipo_cobertura"] = padronizar_texto(df["tipo_cobertura"]).map(mapa_cobertura)

    mapa_sinistro = {
        "colisao": "Colisão", "roubo": "Roubo", "furto": "Furto",
        "incendio": "Incêndio", "danos naturais": "Danos naturais",
    }
    df["tipo_sinistro"] = padronizar_texto(df["tipo_sinistro"]).map(mapa_sinistro)

    nomes_uf = {"bahia": "BA", "sao paulo": "SP"}
    uf = padronizar_texto(df["uf"])
    df["uf"] = uf.replace(nomes_uf).str.upper()

    df["estado_civil"] = padronizar_texto(df["estado_civil"]).str.title().replace({"Viuvo": "Viúvo"})
    df["canal_venda"] = padronizar_texto(df["canal_venda"]).str.title().replace({"Agencia": "Agência"})
    return df


def converter_tipos(df: pd.DataFrame) -> pd.DataFrame:
    df["valor_sinistro"] = df["valor_sinistro"].apply(converter_dinheiro)
    for col in ["data_inicio_apolice", "data_sinistro", "data_aviso"]:
        df[col] = converter_data(df[col])
    return df


def tratar_invalidos_e_outliers(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    # 1) Fora do intervalo possível de idade, tempo de cliente e score -> nulo
    for col in ["idade", "tempo_cliente_anos", "score_credito"]:
        minimo, maximo = LIMITES[col]
        invalidos = (df[col] < minimo) | (df[col] > maximo)
        log[f"invalidos_{col}"] = int(invalidos.sum())
        df.loc[invalidos, col] = np.nan

    # 2) Outliers de valores: limitar ao teto de negócio (winsorização)
    for col in ["renda_mensal", "valor_segurado"]:
        minimo, maximo = LIMITES[col]
        log[f"outliers_{col}"] = int((df[col] > maximo).sum())
        df[col] = df[col].clip(minimo, maximo)

    # 3) Datas sem sentido
    ruins = (df["data_sinistro"] < df["data_inicio_apolice"]) | (df["data_aviso"] < df["data_sinistro"])
    log["datas_incoerentes"] = int(ruins.sum())
    df = df.loc[~ruins].copy()
    return df


def criar_variaveis(df: pd.DataFrame) -> pd.DataFrame:
    # Indicadores de nulo (o modelo pode aprender que "não informar" é sinal)
    for col in COLUNAS_COM_NULOS + ["idade", "tempo_cliente_anos"]:
        df[f"{col}_nulo"] = df[col].isna().astype(int)

    # ---- Apólice x acidente ----
    df["dias_inicio_ate_sinistro"] = (df["data_sinistro"] - df["data_inicio_apolice"]).dt.days
    df["apolice_nova_60d"] = (df["dias_inicio_ate_sinistro"] < 60).astype(int)

    # ---- Aviso ----
    df["atraso_aviso_dias"] = (df["data_aviso"] - df["data_sinistro"]).dt.days
    df["aviso_tardio_10d"] = (df["atraso_aviso_dias"] > 10).astype(int)

    # ---- Valores ----
    df["razao_sinistro_valor_segurado"] = df["valor_sinistro"] / df["valor_segurado"]
    df["razao_sinistro_premio"] = df["valor_sinistro"] / df["premio_anual"]
    df["razao_sinistro_renda"] = df["valor_sinistro"] / df["renda_mensal"]

    # ---- Quando aconteceu ----
    df["sinistro_madrugada"] = df["hora_sinistro"].between(0, 5).astype(int)
    df["sinistro_fim_semana"] = (df["data_sinistro"].dt.dayofweek >= 5).astype(int)
    df["mes_sinistro"] = df["data_sinistro"].dt.month

    # ---- Histórico ----
    df["sinistros_antes_12m"] = (df["qtd_sinistros_total"] - df["qtd_sinistros_12m"]).clip(lower=0)
    df["frequencia_sinistros_ano"] = df["qtd_sinistros_total"] / (df["tempo_cliente_anos"].fillna(0) + 1)
    df["sem_boletim_sem_testemunha"] = (
        (df["boletim_ocorrencia"] == 0) & (df["qtd_testemunhas"].fillna(0) == 0)
    ).astype(int)

    # id_segurado não entra no modelo (identificador), data_aviso/inicio já viraram dias
    df = df.drop(columns=["id_segurado", "data_aviso", "data_inicio_apolice"])
    return df


def relatorio(df_antes: pd.DataFrame, df_depois: pd.DataFrame, log: dict) -> None:
    print("=" * 60)
    print("RELATÓRIO DE QUALIDADE - ETAPA 2")
    print("=" * 60)
    print(f"Linhas iniciais ........... {log['linhas_iniciais']:>7,}")
    print(f"Duplicados removidos ...... {log['duplicados_removidos']:>7,}")
    print(f"Datas incoerentes ......... {log['datas_incoerentes']:>7,}")
    print(f"Idades inválidas -> nulo .. {log['invalidos_idade']:>7,}")
    print(f"Tempo cliente inválido .... {log['invalidos_tempo_cliente_anos']:>7,}")
    print(f"Outliers renda (limitados)  {log['outliers_renda_mensal']:>7,}")
    print(f"Outliers valor segurado ... {log['outliers_valor_segurado']:>7,}")
    print(f"Linhas finais ............. {len(df_depois):>7,}")
    print(f"Colunas: {df_antes.shape[1]} -> {df_depois.shape[1]}")
    print(f"Taxa de fraude ............ {df_depois['fraude'].mean():.2%}")
    print("\nTipos de dados finais:")
    print(df_depois.dtypes.value_counts().to_string())


def main() -> None:
    log: dict = {}
    bruto = pd.read_csv(ENTRADA)
    df = remover_colunas_e_duplicados(bruto, log)
    df = padronizar_categorias(df)
    df = converter_tipos(df)
    df = tratar_invalidos_e_outliers(df, log)
    df = criar_variaveis(df)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAIDA, index=False)
    relatorio(bruto, df, log)
    print(f"\nArquivo salvo em: {SAIDA}")


if __name__ == "__main__":
    main()
