"""Testes das funções de limpeza (Etapa 2)."""
import importlib

import numpy as np
import pandas as pd

pre = importlib.import_module("02_preprocessamento")  # nome começa com número: usa importlib


def test_converter_dinheiro_formato_brasileiro():
    assert pre.converter_dinheiro("R$ 12.345,67") == 12345.67
    assert pre.converter_dinheiro("R$ 900,00") == 900.0


def test_converter_dinheiro_numero_em_texto_e_nulo():
    assert pre.converter_dinheiro("64031.04") == 64031.04
    assert np.isnan(pre.converter_dinheiro(np.nan))


def test_converter_data_aceita_dois_formatos():
    datas = pre.converter_data(pd.Series(["2023-05-17", "17/05/2023"]))
    assert datas.nunique() == 1
    assert datas.iloc[0] == pd.Timestamp("2023-05-17")


def test_padronizar_texto_remove_acento_caixa_e_espaco():
    serie = pd.Series([" COLISÃO ", "colisao", "Colisão"])
    assert padronizado(serie).nunique() == 1


def padronizado(serie):
    return pre.padronizar_texto(serie)


def test_padronizar_categorias_unifica_sinonimos():
    df = pd.DataFrame({
        "boletim_ocorrencia": ["Sim", "SIM", "S", "Não", "NAO", "n"],
        "houve_vitima": ["Sim"] * 6,
        "oficina_credenciada": ["Não"] * 6,
        "tipo_cobertura": ["compreensiva", "COMPREENSIVA", " Compreensiva ", "colisao", "Basica", "roubo e furto"],
        "tipo_sinistro": ["Colisao"] * 6,
        "uf": ["ba", "BA", "Bahia", "SP", "São Paulo", "sp"],
        "estado_civil": ["viuvo"] * 6,
        "canal_venda": ["agencia"] * 6,
    })
    out = pre.padronizar_categorias(df)
    assert out["boletim_ocorrencia"].tolist() == [1, 1, 1, 0, 0, 0]
    assert set(out["uf"]) == {"BA", "SP"}
    assert set(out["tipo_cobertura"]) == {"Compreensiva", "Colisão", "Básica", "Roubo e Furto"}
    assert out["estado_civil"].iloc[0] == "Viúvo"


def test_invalidos_viram_nulo_e_outliers_sao_limitados():
    df = pd.DataFrame({
        "idade": [30, -1, 150], "tempo_cliente_anos": [2.0, -1, 5.0], "score_credito": [600, 700, 5000],
        "renda_mensal": [3000.0, 999999.0, 100.0], "valor_segurado": [50000.0, 9e6, 1000.0],
        "data_inicio_apolice": pd.to_datetime(["2023-01-01"] * 3),
        "data_sinistro": pd.to_datetime(["2023-02-01"] * 3),
        "data_aviso": pd.to_datetime(["2023-02-02"] * 3),
    })
    log = {}
    out = pre.tratar_invalidos_e_outliers(df, log)
    assert out["idade"].isna().sum() == 2
    assert out["tempo_cliente_anos"].isna().sum() == 1
    assert out["score_credito"].isna().sum() == 1
    assert out["renda_mensal"].max() == 60_000
    assert out["renda_mensal"].min() == 1_500
    assert out["valor_segurado"].max() == 400_000


def test_datas_incoerentes_sao_removidas():
    df = pd.DataFrame({
        "idade": [30, 30], "tempo_cliente_anos": [1.0, 1.0], "score_credito": [600, 600],
        "renda_mensal": [3000.0, 3000.0], "valor_segurado": [50000.0, 50000.0],
        "data_inicio_apolice": pd.to_datetime(["2023-01-01", "2023-03-01"]),
        "data_sinistro": pd.to_datetime(["2023-02-01", "2023-02-01"]),  # 2ª linha: sinistro antes da apólice
        "data_aviso": pd.to_datetime(["2023-02-02", "2023-02-02"]),
    })
    out = pre.tratar_invalidos_e_outliers(df, {})
    assert len(out) == 1
