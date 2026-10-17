"""Testes das regras de negócio, priorização, divisão temporal e PSI (src/comum.py)."""
import numpy as np
import pandas as pd

from comum import (
    ParametrosNegocio, calcular_psi, comparar_estrategias, dividir_temporal, distribuir_analistas,
    metricas_negocio, selecionar_casos, status_psi, valor_esperado_liquido,
)

P = ParametrosNegocio(custo_investigacao=1000.0, taxa_prevencao=0.5, capacidade_pct=0.5)


def test_valor_esperado_liquido_formula():
    # 0,5 x 10.000 x 0,5 - 1.000 = 1.500
    assert abs(float(valor_esperado_liquido(0.5, 10_000, P)) - 1500.0) < 1e-9


def test_selecao_respeita_capacidade_e_ignora_prejuizo():
    prob = np.array([0.9, 0.8, 0.01, 0.02])
    valor = np.array([10_000, 10_000, 10_000, 10_000])
    mascara = selecionar_casos(prob, valor, P)  # capacidade 50% de 4 = 2 casos
    assert mascara.tolist() == [True, True, False, False]
    # com custo altíssimo, nenhum caso compensa
    caro = ParametrosNegocio(custo_investigacao=1e9, taxa_prevencao=0.5, capacidade_pct=0.5)
    assert selecionar_casos(prob, valor, caro).sum() == 0


def test_metricas_negocio_exemplo_simples():
    y = np.array([1, 1, 0, 0])
    prob = np.array([0.9, 0.8, 0.1, 0.05])
    valor = np.array([10_000, 20_000, 10_000, 10_000])
    m = metricas_negocio(y, prob, valor, P)
    assert m["casos_investigados"] == 2
    assert m["fraudes_capturadas"] == 2
    assert abs(m["valor_protegido"] - (10_000 + 20_000) * 0.5) < 1e-9
    assert abs(m["resultado_liquido"] - (15_000 - 2_000)) < 1e-9
    assert m["recall_na_capacidade"] == 1.0


def test_modelo_informativo_vence_aleatorio():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.07).astype(int)
    prob = np.clip(0.05 + 0.6 * y + rng.normal(0, 0.05, 2000), 0.001, 0.99)
    valor = rng.uniform(5_000, 50_000, 2000)
    t = comparar_estrategias(y, prob, valor, ParametrosNegocio(capacidade_pct=0.10)).set_index("estrategia")
    assert t.loc["Modelo + valor esperado", "resultado_liquido"] > t.loc["Aleatório (média)", "resultado_liquido"]
    assert t.loc["Não investigar", "resultado_liquido"] == 0


def test_psi_igual_e_deslocado():
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(0, 1, 5000))
    assert calcular_psi(base, pd.Series(rng.normal(0, 1, 5000))) < 0.05
    assert calcular_psi(base, pd.Series(rng.normal(2, 1, 5000))) > 0.25
    assert status_psi(0.05) == "estável" and status_psi(0.15) == "atenção" and status_psi(0.4) == "alerta"


def test_psi_categorias():
    ref = pd.Series(["a"] * 80 + ["b"] * 20)
    assert calcular_psi(ref, pd.Series(["a"] * 80 + ["b"] * 20)) < 0.01
    assert calcular_psi(ref, pd.Series(["a"] * 20 + ["b"] * 80)) > 0.25


def test_distribuicao_em_serpentina_equilibra_carga():
    nomes = distribuir_analistas(pd.Series(range(10)), 3)
    assert nomes[:6] == ["Analista 1", "Analista 2", "Analista 3", "Analista 3", "Analista 2", "Analista 1"]
    assert pd.Series(nomes).value_counts().max() - pd.Series(nomes).value_counts().min() <= 1


def test_divisao_temporal_nao_mistura_passado_e_futuro():
    df = pd.DataFrame({
        "id_sinistro": [f"S{i}" for i in range(100)],
        "data_sinistro": pd.date_range("2023-01-01", periods=100)[::-1],  # embaralhado em ordem inversa
    })
    treino, valid, teste = dividir_temporal(df)
    assert len(treino) == 60 and len(valid) == 20 and len(teste) == 20
    assert treino["data_sinistro"].max() <= valid["data_sinistro"].min()
    assert valid["data_sinistro"].max() <= teste["data_sinistro"].min()
