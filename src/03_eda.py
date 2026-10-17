"""
ETAPA 3 - Análise exploratória (EDA) com gráficos.

Responde: onde a fraude aparece mais? Quais sinais separam fraude de sinistro legítimo?

Como rodar
----------
    python src/03_eda.py

Entrada : data/processed/sinistros_tratado.csv
Saída   : reports/figuras/eda_*.png  e  reports/eda_resumo.json
"""
import json

import matplotlib

matplotlib.use("Agg")  # gera imagens sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from comum import ALVO, COLS_NUM, PASTA_FIGURAS, PASTA_REPORTS, carregar_tratado

AZUL, LARANJA, CINZA = "#2F6DB5", "#E8743B", "#8A8A8A"
sns.set_theme(style="whitegrid", font_scale=0.95)


def salvar(fig, nome: str) -> None:
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / nome, dpi=130)
    plt.close(fig)
    print(f"  figura salva: {nome}")


def barras_taxa(ax, df, coluna, titulo, ordem=None):
    """Barras com a taxa de fraude de cada categoria."""
    g = df.groupby(coluna)[ALVO].agg(["mean", "size"])
    if ordem is not None:
        g = g.reindex(ordem)
    else:
        g = g.sort_values("mean", ascending=False)
    ax.bar(g.index.astype(str), g["mean"] * 100, color=AZUL)
    ax.axhline(df[ALVO].mean() * 100, color=LARANJA, ls="--", lw=1.5, label="média geral")
    for i, (taxa, n) in enumerate(zip(g["mean"], g["size"])):
        ax.text(i, taxa * 100 + 0.15, f"{taxa:.1%}", ha="center", fontsize=8)
    ax.set_title(titulo, fontsize=10)
    ax.set_ylabel("% de fraude")
    ax.tick_params(axis="x", rotation=30)


def grafico_desbalanceamento(df):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    contagem = df[ALVO].value_counts().sort_index()
    ax.bar(["Legítimo", "Fraude"], contagem.values, color=[AZUL, LARANJA])
    for i, v in enumerate(contagem.values):
        ax.text(i, v + 200, f"{v:,}\n({v / len(df):.1%})", ha="center")
    ax.set_title("Classes desbalanceadas")
    ax.set_ylabel("Sinistros")
    ax.set_ylim(0, contagem.max() * 1.18)
    salvar(fig, "eda_01_desbalanceamento.png")


def grafico_categorias(df):
    fig, eixos = plt.subplots(2, 3, figsize=(14, 7.5))
    barras_taxa(eixos[0, 0], df, "tipo_sinistro", "Por tipo de sinistro")
    barras_taxa(eixos[0, 1], df, "tipo_cobertura", "Por cobertura")
    barras_taxa(eixos[0, 2], df, "canal_venda", "Por canal de venda")
    barras_taxa(eixos[1, 0], df.assign(b=df["boletim_ocorrencia"].map({1: "Com B.O.", 0: "Sem B.O."})),
                "b", "Boletim de ocorrência")
    barras_taxa(eixos[1, 1], df.assign(o=df["oficina_credenciada"].map({1: "Credenciada", 0: "Não credenciada"})),
                "o", "Oficina")
    barras_taxa(eixos[1, 2], df, "uf", "Por estado")
    fig.suptitle("Taxa de fraude por categoria", fontsize=13)
    salvar(fig, "eda_02_taxa_por_categoria.png")


def grafico_tempo_apolice(df):
    faixas = pd.cut(df["dias_inicio_ate_sinistro"], [0, 30, 60, 120, 365, 2000],
                    labels=["até 30 dias", "31–60", "61–120", "121–365", "mais de 1 ano"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    barras_taxa(ax, df.assign(faixa=faixas), "faixa", "Fraude x tempo entre contratar e sinistrar",
                ordem=faixas.cat.categories)
    salvar(fig, "eda_03_tempo_apolice.png")


def grafico_razao_valor(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for valor, nome, cor in [(0, "Legítimo", AZUL), (1, "Fraude", LARANJA)]:
        sns.kdeplot(df.loc[df[ALVO] == valor, "razao_sinistro_valor_segurado"], ax=ax,
                    label=nome, color=cor, fill=True, alpha=0.3, common_norm=False)
    ax.set_title("Valor do sinistro ÷ valor segurado")
    ax.set_xlabel("Razão (1,0 = perda total)")
    ax.legend()
    salvar(fig, "eda_04_razao_valor.png")


def grafico_hora(df):
    por_hora = df.groupby("hora_sinistro")[ALVO].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(por_hora.index, por_hora.values, marker="o", color=AZUL)
    ax.axhline(df[ALVO].mean() * 100, color=LARANJA, ls="--", label="média geral")
    ax.axvspan(-0.5, 5.5, color=CINZA, alpha=0.15, label="madrugada")
    ax.set_xlabel("Hora do sinistro")
    ax.set_ylabel("% de fraude")
    ax.set_title("Fraude por horário")
    ax.legend()
    salvar(fig, "eda_05_hora.png")


def grafico_aviso_e_historico(df):
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.3))
    atraso = pd.cut(df["atraso_aviso_dias"], [-1, 0, 3, 10, 100], labels=["mesmo dia", "1–3 dias", "4–10", "mais de 10"])
    barras_taxa(eixos[0], df.assign(f=atraso), "f", "Atraso para avisar", ordem=atraso.cat.categories)
    hist = df["qtd_sinistros_12m"].clip(upper=4).astype(int).map(lambda v: "4 ou mais" if v == 4 else str(v))
    barras_taxa(eixos[1], df.assign(h=hist), "h", "Sinistros nos últimos 12 meses",
                ordem=["0", "1", "2", "3", "4 ou mais"])
    salvar(fig, "eda_06_aviso_historico.png")


def grafico_valor_boxplot(df):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.boxplot(data=df, x=ALVO, y="valor_sinistro", hue=ALVO, palette=[AZUL, LARANJA], legend=False, ax=ax)
    ax.set_yscale("log")
    ax.set_xticks([0, 1], ["Legítimo", "Fraude"])
    ax.set_xlabel("")
    ax.set_ylabel("Valor do sinistro (R$, escala log)")
    ax.set_title("Valor do sinistro")
    salvar(fig, "eda_07_valor_boxplot.png")


def grafico_correlacao(df):
    corr = df[COLS_NUM + [ALVO]].corr(numeric_only=True)[ALVO].drop(ALVO)
    top = corr.reindex(corr.abs().sort_values(ascending=False).head(15).index).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top.index, top.values, color=[LARANJA if v > 0 else AZUL for v in top.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Variáveis mais ligadas à fraude (correlação)")
    ax.set_xlabel("Correlação com fraude")
    salvar(fig, "eda_08_correlacao.png")
    return corr


def grafico_nulos(df):
    nulos = (df.isna().mean() * 100).sort_values(ascending=False)
    nulos = nulos[nulos > 0].head(10)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(nulos.index[::-1], nulos.values[::-1], color=CINZA)
    ax.set_xlabel("% de valores vazios")
    ax.set_title("Valores nulos que seguem para o Pipeline")
    salvar(fig, "eda_09_nulos.png")


def main() -> None:
    df = carregar_tratado()
    print(f"Base tratada: {len(df):,} linhas | fraude = {df[ALVO].mean():.2%}")
    grafico_desbalanceamento(df)
    grafico_categorias(df)
    grafico_tempo_apolice(df)
    grafico_razao_valor(df)
    grafico_hora(df)
    grafico_aviso_e_historico(df)
    grafico_valor_boxplot(df)
    corr = grafico_correlacao(df)
    grafico_nulos(df)

    resumo = {
        "linhas": int(len(df)),
        "taxa_fraude": float(df[ALVO].mean()),
        "taxa_fraude_apolice_nova_60d": float(df.loc[df["apolice_nova_60d"] == 1, ALVO].mean()),
        "taxa_fraude_apolice_antiga": float(df.loc[df["apolice_nova_60d"] == 0, ALVO].mean()),
        "taxa_fraude_sem_bo": float(df.loc[df["boletim_ocorrencia"] == 0, ALVO].mean()),
        "taxa_fraude_com_bo": float(df.loc[df["boletim_ocorrencia"] == 1, ALVO].mean()),
        "taxa_fraude_madrugada": float(df.loc[df["sinistro_madrugada"] == 1, ALVO].mean()),
        "taxa_fraude_aviso_tardio": float(df.loc[df["aviso_tardio_10d"] == 1, ALVO].mean()),
        "taxa_fraude_razao_acima_45": float(df.loc[df["razao_sinistro_valor_segurado"] > 0.45, ALVO].mean()),
        "taxa_fraude_razao_ate_45": float(df.loc[df["razao_sinistro_valor_segurado"] <= 0.45, ALVO].mean()),
        "valor_medio_sinistro_fraude": float(df.loc[df[ALVO] == 1, "valor_sinistro"].mean()),
        "valor_medio_sinistro_legitimo": float(df.loc[df[ALVO] == 0, "valor_sinistro"].mean()),
        "top_correlacoes": corr.reindex(corr.abs().sort_values(ascending=False).head(8).index).round(3).to_dict(),
    }
    PASTA_REPORTS.mkdir(exist_ok=True)
    (PASTA_REPORTS / "eda_resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
