"""
ETAPA 4 - Modelagem: 3 modelos, métricas de negócio e explicabilidade.

Modelos (scikit-learn): Regressão Logística, Random Forest e Gradient Boosting.

O que este arquivo faz
----------------------
1. Divide os dados no tempo: treino (passado) / validação / teste (futuro)
2. Treina os 3 modelos dentro de um Pipeline (imputação + escala + one-hot + calibração)
3. Avalia com métricas estatísticas E de negócio (valor protegido - custo)
4. Escolhe o melhor modelo pelo resultado líquido na VALIDAÇÃO (o teste fica só para a nota final)
5. Explica os modelos (importância por permutação)
6. Salva modelos, métricas, previsões e gráficos

Como rodar
----------
    python src/04_modelagem.py
"""
import json
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, roc_curve

from comum import (
    ALVO, COL_DATA, COL_ID, FEATURES, MODELO_FINAL, METRICAS_JSON, PASTA_FIGURAS, PASTA_MODELOS,
    PASTA_REPORTS, PREDICOES, ParametrosNegocio, carregar_tratado, construir_pipeline,
    criar_modelos, curva_capacidade, dividir_temporal, metricas_negocio,
)

warnings.filterwarnings("ignore")
CORES = {"regressao_logistica": "#2F6DB5", "random_forest": "#3FA34D", "gradient_boosting": "#E8743B"}
NOMES_BONITOS = {
    "regressao_logistica": "Regressão Logística",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}
COLUNAS_EXIBIR = [
    "tipo_sinistro", "uf", "boletim_ocorrencia", "dias_inicio_ate_sinistro",
    "atraso_aviso_dias", "razao_sinistro_valor_segurado",
]


def salvar(fig, nome):
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / nome, dpi=130)
    plt.close(fig)
    print(f"  figura salva: {nome}")


def grafico_roc_pr(teste, probs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for nome, prob in probs.items():
        fpr, tpr, _ = roc_curve(teste[ALVO], prob)
        prec, rec, _ = precision_recall_curve(teste[ALVO], prob)
        ax1.plot(fpr, tpr, label=NOMES_BONITOS[nome], color=CORES[nome])
        ax2.plot(rec, prec, label=NOMES_BONITOS[nome], color=CORES[nome])
    ax1.plot([0, 1], [0, 1], "--", color="gray")
    ax1.set(title="Curva ROC (teste)", xlabel="Falsos positivos", ylabel="Fraudes detectadas")
    ax2.axhline(teste[ALVO].mean(), ls="--", color="gray", label="sem modelo")
    ax2.set(title="Precisão x Recall (teste)", xlabel="Recall", ylabel="Precisão")
    ax1.legend()
    ax2.legend()
    salvar(fig, "modelo_01_roc_pr.png")


def grafico_resultado_liquido(tabela):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivo = tabela.pivot(index="modelo", columns="conjunto", values="resultado_liquido") / 1e6
    pivo = pivo.reindex(list(NOMES_BONITOS))
    x = np.arange(len(pivo))
    ax.bar(x - 0.2, pivo["validacao"], 0.4, label="Validação", color="#8FB4E3")
    ax.bar(x + 0.2, pivo["teste"], 0.4, label="Teste", color="#2F6DB5")
    for i, (v, t) in enumerate(zip(pivo["validacao"], pivo["teste"])):
        ax.text(i - 0.2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        ax.text(i + 0.2, t + 0.02, f"{t:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, [NOMES_BONITOS[m] for m in pivo.index])
    ax.set_ylabel("Resultado líquido (R$ milhões)")
    ax.set_title("Valor protegido − custo de investigação")
    ax.legend()
    salvar(fig, "modelo_02_resultado_liquido.png")


def grafico_calibracao(teste, probs):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for nome, prob in probs.items():
        freq, media = calibration_curve(teste[ALVO], prob, n_bins=8, strategy="quantile")
        ax.plot(media, freq, marker="o", label=NOMES_BONITOS[nome], color=CORES[nome])
    ax.plot([0, 0.5], [0, 0.5], "--", color="gray", label="perfeita")
    ax.set(title="Calibração das probabilidades (teste)", xlabel="Probabilidade prevista", ylabel="Taxa real de fraude")
    ax.legend()
    salvar(fig, "modelo_03_calibracao.png")


def grafico_ganho_acumulado(teste, probs):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    n = len(teste)
    y = teste[ALVO].values
    for nome, prob in probs.items():
        ordem = np.argsort(-prob)
        acumulado = np.cumsum(y[ordem]) / y.sum()
        ax.plot(np.arange(1, n + 1) / n * 100, acumulado * 100, label=NOMES_BONITOS[nome], color=CORES[nome])
    ax.plot([0, 100], [0, 100], "--", color="gray", label="aleatório")
    ax.set(title="Ganho acumulado (teste)", xlabel="% dos sinistros investigados (do mais ao menos arriscado)",
           ylabel="% das fraudes encontradas", xlim=(0, 100), ylim=(0, 100))
    ax.legend()
    salvar(fig, "modelo_04_ganho_acumulado.png")


def grafico_importancias(importancias):
    fig, eixos = plt.subplots(1, 3, figsize=(16, 5.5), sharex=False)
    for ax, (nome, imp) in zip(eixos, importancias.items()):
        top = imp.head(12).iloc[::-1]
        ax.barh(top["variavel"], top["importancia"], xerr=top["desvio"], color=CORES[nome])
        ax.set_title(NOMES_BONITOS[nome])
        ax.set_xlabel("Queda no PR-AUC ao embaralhar a variável")
    fig.suptitle("Explicabilidade global: importância por permutação (teste)", fontsize=13)
    salvar(fig, "modelo_05_importancia.png")


def importancia_permutacao(pipe, teste):
    amostra = teste.sample(n=min(3000, len(teste)), random_state=42)
    r = permutation_importance(
        pipe, amostra[FEATURES], amostra[ALVO], scoring="average_precision",
        n_repeats=3, random_state=42, n_jobs=-1,
    )
    return (
        pd.DataFrame({"variavel": FEATURES, "importancia": r.importances_mean, "desvio": r.importances_std})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    p = ParametrosNegocio()
    df = carregar_tratado()
    treino, valid, teste = dividir_temporal(df)
    print(f"Treino: {len(treino):,} | Validação: {len(valid):,} | Teste: {len(teste):,}")
    print(f"Taxa de fraude -> treino {treino[ALVO].mean():.2%} | valid {valid[ALVO].mean():.2%} | teste {teste[ALVO].mean():.2%}")
    print(f"Premissas: custo R$ {p.custo_investigacao:,.0f} | prevenção {p.taxa_prevencao:.0%} | capacidade {p.capacidade_pct:.0%}\n")

    PASTA_MODELOS.mkdir(exist_ok=True)
    resultados, probs_teste, probs_valid, modelos = [], {}, {}, {}

    for nome, estimador in criar_modelos().items():
        t0 = time.time()
        pipe = construir_pipeline(estimador)
        pipe.fit(treino[FEATURES], treino[ALVO])
        modelos[nome] = pipe
        for conjunto, dados in (("validacao", valid), ("teste", teste)):
            prob = pipe.predict_proba(dados[FEATURES])[:, 1]
            (probs_valid if conjunto == "validacao" else probs_teste)[nome] = prob
            m = metricas_negocio(dados[ALVO].values, prob, dados["valor_sinistro"].values, p)
            resultados.append({"modelo": nome, "conjunto": conjunto, **m})
        joblib.dump(pipe, PASTA_MODELOS / f"{nome}.joblib", compress=3)
        r = [x for x in resultados if x["modelo"] == nome]
        print(f"{NOMES_BONITOS[nome]:<20} {time.time() - t0:5.1f}s | AUC teste {r[1]['roc_auc']:.3f} | "
              f"PR-AUC {r[1]['pr_auc']:.3f} | líquido teste R$ {r[1]['resultado_liquido']:,.0f}")

    tabela = pd.DataFrame(resultados)

    # Seleção do melhor modelo: resultado líquido na VALIDAÇÃO
    validacao = tabela[tabela["conjunto"] == "validacao"].set_index("modelo")
    melhor = validacao["resultado_liquido"].idxmax()
    print(f"\n>>> Melhor modelo (validação, resultado líquido): {NOMES_BONITOS[melhor]}")
    joblib.dump(modelos[melhor], MODELO_FINAL, compress=3)

    # Explicabilidade global
    print("\nCalculando importância por permutação...")
    importancias = {nome: importancia_permutacao(pipe, teste) for nome, pipe in modelos.items()}
    PASTA_REPORTS.mkdir(exist_ok=True)
    pd.concat([imp.assign(modelo=nome) for nome, imp in importancias.items()]).to_csv(
        PASTA_REPORTS / "importancia_permutacao.csv", index=False)

    # Curva de capacidade do melhor modelo (teste)
    curva = curva_capacidade(teste[ALVO].values, probs_teste[melhor], teste["valor_sinistro"].values, p)
    curva.to_csv(PASTA_REPORTS / "curva_capacidade_teste.csv", index=False)

    # Previsões do teste (alimentam priorização, monitoramento e app)
    pred = teste[[COL_ID, COL_DATA, "valor_sinistro", ALVO] + COLUNAS_EXIBIR].copy()
    for nome, prob in probs_teste.items():
        pred[f"prob_{nome}"] = prob
    pred["prob_modelo"] = probs_teste[melhor]
    pred.to_csv(PREDICOES, index=False)

    # Gráficos
    grafico_roc_pr(teste, probs_teste)
    grafico_resultado_liquido(tabela)
    grafico_calibracao(teste, probs_teste)
    grafico_ganho_acumulado(teste, probs_teste)
    grafico_importancias(importancias)

    # Métricas em JSON (o app e o MLflow leem este arquivo)
    saida = {
        "parametros_negocio": p.como_dict(),
        "divisao": {
            "treino": {"linhas": len(treino), "fim": str(treino[COL_DATA].max().date())},
            "validacao": {"linhas": len(valid), "fim": str(valid[COL_DATA].max().date())},
            "teste": {"linhas": len(teste), "fim": str(teste[COL_DATA].max().date())},
        },
        "melhor_modelo": melhor,
        "resultados": resultados,
        "top_variaveis_melhor_modelo": importancias[melhor].head(10)[["variavel", "importancia"]].round(5).to_dict("records"),
    }
    METRICAS_JSON.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")

    cols = ["modelo", "conjunto", "roc_auc", "pr_auc", "recall_na_capacidade", "precisao_na_capacidade",
            "lift_na_capacidade", "valor_protegido", "custo_investigacao", "resultado_liquido", "roi"]
    print("\n", tabela[cols].round(3).to_string(index=False))
    print("\nTop variáveis (melhor modelo):")
    print(importancias[melhor].head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
