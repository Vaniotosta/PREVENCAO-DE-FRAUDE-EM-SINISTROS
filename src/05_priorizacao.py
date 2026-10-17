"""
ETAPA 5 - Priorização da fila de investigação.

Transforma a probabilidade de fraude em uma decisão de trabalho:
  valor esperado líquido = prob. de fraude x valor do sinistro x taxa de prevenção - custo
Ordena do maior para o menor, respeita a capacidade da equipe e divide entre os analistas.

Como rodar
----------
    python src/05_priorizacao.py [--capacidade 0.10] [--analistas 5]

Saída
-----
    reports/fila_priorizada.csv           fila com motivos, prioridade e analista
    reports/comparacao_estrategias.csv    modelo x alternativas sem modelo
    data/processed/motivos_teste.csv      motivos de alerta de todos os casos do teste
    reports/figuras/prioridade_*.png
"""
import argparse

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comum import (
    ALVO, COL_ID, FEATURES, MODELO_FINAL, MOTIVOS, PASTA_FIGURAS, PASTA_REPORTS, PREDICOES,
    ParametrosNegocio, carregar_tratado, comparar_estrategias, curva_capacidade, dividir_temporal,
    distribuir_analistas, explicar_lote, principais_motivos, rotular_prioridade,
    selecionar_casos, valor_esperado_liquido, valores_tipicos,
)


def salvar(fig, nome):
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / nome, dpi=130)
    plt.close(fig)
    print(f"  figura salva: {nome}")


def grafico_estrategias(tabela, capacidade):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    t = tabela[~tabela["estrategia"].isin(["Ideal (só fraudes, impossível)"])].copy()
    cores = ["#8A8A8A" if e != "Modelo + valor esperado" else "#2F6DB5" for e in t["estrategia"]]
    ax.barh(t["estrategia"], t["resultado_liquido"] / 1e6, color=cores)
    for i, v in enumerate(t["resultado_liquido"] / 1e6):
        ax.text(v + (0.03 if v >= 0 else -0.03), i, f"R$ {v:.2f} mi", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Resultado líquido (R$ milhões)")
    ax.set_title(f"Modelo x alternativas sem modelo (capacidade {capacidade:.0%}, base de teste)")
    ax.invert_yaxis()
    salvar(fig, "prioridade_01_estrategias.png")


def grafico_capacidade(curva, capacidade):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curva["capacidade_pct"] * 100, curva["resultado_liquido"] / 1e6, color="#2F6DB5", lw=2)
    melhor = curva.loc[curva["resultado_liquido"].idxmax()]
    ax.scatter([melhor["capacidade_pct"] * 100], [melhor["resultado_liquido"] / 1e6], color="#E8743B", zorder=3,
               label=f"melhor ponto: {melhor['capacidade_pct']:.0%}")
    ax.axvline(capacidade * 100, color="gray", ls="--", label=f"capacidade atual: {capacidade:.0%}")
    ax.set_xlabel("% dos sinistros investigados")
    ax.set_ylabel("Resultado líquido (R$ milhões)")
    ax.set_title("Quanto vale aumentar a equipe de investigação?")
    ax.legend()
    salvar(fig, "prioridade_02_curva_capacidade.png")
    return melhor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacidade", type=float, default=0.10, help="fração de sinistros que a equipe investiga")
    parser.add_argument("--analistas", type=int, default=5, help="número de analistas")
    args = parser.parse_args()

    p = ParametrosNegocio(capacidade_pct=args.capacidade)
    pred = pd.read_csv(PREDICOES, parse_dates=["data_sinistro"])
    modelo = joblib.load(MODELO_FINAL)

    # --- Motivos de alerta para todos os casos do teste (explicação por grupos) ---
    df = carregar_tratado()
    treino, _, teste = dividir_temporal(df)
    teste = teste.set_index(COL_ID).loc[pred[COL_ID]].reset_index()
    contribuicoes = explicar_lote(modelo, teste, valores_tipicos(treino))
    motivos = pd.concat([pred[[COL_ID]].reset_index(drop=True), principais_motivos(contribuicoes)], axis=1)
    motivos.to_csv(MOTIVOS, index=False)

    # --- Fila ---
    pred["valor_esperado_liquido"] = valor_esperado_liquido(pred["prob_modelo"], pred["valor_sinistro"], p)
    pred["valor_protegido_esperado"] = pred["prob_modelo"] * pred["valor_sinistro"] * p.taxa_prevencao
    mascara = selecionar_casos(pred["prob_modelo"].values, pred["valor_sinistro"].values, p)
    fila = pred.loc[mascara].merge(motivos, on=COL_ID).sort_values("valor_esperado_liquido", ascending=False)
    fila.insert(0, "posicao", np.arange(1, len(fila) + 1))
    fila["prioridade"] = rotular_prioridade(fila["posicao"], len(fila))
    fila["analista"] = distribuir_analistas(fila["posicao"], args.analistas)
    colunas = ["posicao", "prioridade", "analista", COL_ID, "prob_modelo", "valor_sinistro",
               "valor_protegido_esperado", "valor_esperado_liquido", "motivo_principal", "motivo_secundario",
               "tipo_sinistro", "uf", ALVO]
    PASTA_REPORTS.mkdir(exist_ok=True)
    fila[colunas].rename(columns={"prob_modelo": "prob_fraude", ALVO: "fraude_real_(so_para_avaliar)"}).to_csv(
        PASTA_REPORTS / "fila_priorizada.csv", index=False)

    # --- Comparação com alternativas e curva de capacidade ---
    tabela = comparar_estrategias(pred[ALVO].values, pred["prob_modelo"].values, pred["valor_sinistro"].values, p)
    tabela.to_csv(PASTA_REPORTS / "comparacao_estrategias.csv", index=False)
    curva = curva_capacidade(pred[ALVO].values, pred["prob_modelo"].values, pred["valor_sinistro"].values, p)
    grafico_estrategias(tabela, args.capacidade)
    melhor = grafico_capacidade(curva, args.capacidade)

    # --- Resumo ---
    print(f"\nFila: {len(fila):,} casos de {len(pred):,} ({len(fila) / len(pred):.1%}) | analistas: {args.analistas}")
    print(f"Fraudes na fila: {int(fila[ALVO].sum())} de {int(pred[ALVO].sum())} "
          f"(precisão {fila[ALVO].mean():.1%}, base {pred[ALVO].mean():.1%})")
    print("\nCarga por analista:")
    print(fila.groupby("analista").agg(casos=(COL_ID, "count"),
                                       valor_esperado=("valor_esperado_liquido", "sum")).round(0).to_string())
    print("\nComparação de estratégias:")
    print(tabela.round(0).to_string(index=False))
    print(f"\nMelhor capacidade na curva: {melhor['capacidade_pct']:.0%} "
          f"(resultado R$ {melhor['resultado_liquido']:,.0f})")
    print("\nTop 5 da fila:")
    print(fila[["posicao", "prioridade", "analista", COL_ID, "prob_modelo", "valor_sinistro",
                "motivo_principal"]].head(5).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
