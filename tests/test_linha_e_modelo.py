"""Garante que o app calcula as variáveis igual ao pré-processamento e que o modelo funciona."""
import numpy as np
import pandas as pd

from comum import (
    COLS_COM_INDICADOR_NULO, COL_DATA, FEATURES, MODELO_FINAL, carregar_tratado,
    construir_linha, explicar_lote, dividir_temporal, valores_tipicos,
)

CAMPOS_ENTRADA = [
    "idade", "tempo_cliente_anos", "score_credito", "renda_mensal", "estado_civil", "tipo_cobertura",
    "valor_segurado", "franquia", "premio_anual", "canal_venda", "alteracoes_apolice_90d",
    "qtd_sinistros_12m", "qtd_sinistros_total", "dias_inicio_ate_sinistro", "atraso_aviso_dias",
    "tipo_sinistro", "hora_sinistro", "uf", "boletim_ocorrencia", "qtd_testemunhas", "houve_vitima",
    "oficina_credenciada", "valor_sinistro",
]


def test_todas_as_variaveis_existem_na_base_tratada():
    df = carregar_tratado()
    assert set(FEATURES) <= set(df.columns)


def test_construir_linha_replica_o_preprocessamento():
    df = carregar_tratado().sample(300, random_state=3)
    for _, linha in df.iterrows():
        entrada = {c: (None if pd.isna(linha[c]) else linha[c]) for c in CAMPOS_ENTRADA}
        entrada["dia_semana"] = linha[COL_DATA].dayofweek
        nova = construir_linha(entrada).iloc[0]
        for col in FEATURES:
            esperado, obtido = linha[col], nova[col]
            if isinstance(esperado, str):
                assert esperado == obtido, col
            else:
                assert np.isclose(float(esperado), float(obtido), equal_nan=True), (col, esperado, obtido)


def test_nulo_vira_indicador():
    df = carregar_tratado().iloc[[0]]
    entrada = {c: df.iloc[0][c] for c in CAMPOS_ENTRADA}
    entrada["dia_semana"] = 2
    entrada["score_credito"] = None
    linha = construir_linha(entrada)
    assert linha["score_credito_nulo"].iloc[0] == 1
    assert np.isnan(linha["score_credito"].iloc[0])
    assert set(f"{c}_nulo" for c in COLS_COM_INDICADOR_NULO) <= set(linha.columns)


def test_modelo_final_gera_probabilidades_validas_e_explicacao():
    if not MODELO_FINAL.exists():
        return  # rode a Etapa 4 antes
    import joblib

    modelo = joblib.load(MODELO_FINAL)
    treino, _, teste = dividir_temporal(carregar_tratado())
    amostra = teste.head(50)
    prob = modelo.predict_proba(amostra[FEATURES])[:, 1]
    assert prob.shape == (50,) and ((prob >= 0) & (prob <= 1)).all()
    contrib = explicar_lote(modelo, amostra, valores_tipicos(treino))
    assert contrib.shape[0] == 50 and contrib.shape[1] >= 10
