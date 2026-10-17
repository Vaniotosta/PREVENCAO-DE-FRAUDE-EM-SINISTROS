"""
ETAPA 7 - Monitoramento do modelo.

Compara o dado "de referência" (treino) com o dado "atual" (teste = o futuro) e responde:
  1. As variáveis mudaram? (PSI - Population Stability Index)
  2. As notas de risco mudaram? (PSI do score)
  3. O desempenho caiu? (AUC, PR-AUC e resultado líquido contra a validação)

Regras de alerta
----------------
PSI < 0,10 estável | 0,10 a 0,25 atenção | acima de 0,25 alerta (revisar / retreinar)

Como rodar
----------
    python src/07_monitoramento.py                    # dados reais de teste
    python src/07_monitoramento.py --simular-drift    # simula um cenário de mudança (demonstração)

Saída
-----
    reports/monitoramento[_simulado].csv / .json / figura
    (e registro no MLflow, se estiver instalado)
"""
import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comum import (
    ALVO, COLS_NUM, FEATURES, METRICAS_JSON, MLFLOW_MONITORAMENTO, MLFLOW_URI, MODELO_FINAL,
    PASTA_FIGURAS, PASTA_REPORTS, ParametrosNegocio, calcular_psi, carregar_tratado,
    dividir_temporal, metricas_negocio, status_psi,
)

LIMITE_QUEDA_AUC = 0.05          # queda de AUC (absoluta) que dispara alerta
LIMITE_QUEDA_RESULTADO = 0.25    # queda de 25% no resultado líquido dispara alerta


def simular_drift(df: pd.DataFrame, semente: int = 7) -> pd.DataFrame:
    """Cenário fictício: sinistros mais caros, avisos mais demorados, mais ocorrências de madrugada."""
    rng = np.random.default_rng(semente)
    d = df.copy()
    d["valor_sinistro"] *= 1.5
    d["razao_sinistro_valor_segurado"] = (d["valor_sinistro"] / d["valor_segurado"]).clip(upper=1.0)
    d["razao_sinistro_premio"] = d["valor_sinistro"] / d["premio_anual"]
    d["razao_sinistro_renda"] = d["valor_sinistro"] / d["renda_mensal"]
    d["atraso_aviso_dias"] = d["atraso_aviso_dias"] + 4
    d["aviso_tardio_10d"] = (d["atraso_aviso_dias"] > 10).astype(int)
    mover = rng.random(len(d)) < 0.25
    d.loc[mover, "hora_sinistro"] = rng.integers(0, 6, mover.sum())
    d["sinistro_madrugada"] = d["hora_sinistro"].between(0, 5).astype(int)
    d["score_credito"] = (d["score_credito"] - 80).clip(lower=250)
    return d


def grafico_psi(tabela: pd.DataFrame, sufixo: str) -> None:
    top = tabela.sort_values("psi", ascending=False).head(15).iloc[::-1]
    cores = top["status"].map({"estável": "#3FA34D", "atenção": "#E8B93B", "alerta": "#D64545"})
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["variavel"], top["psi"], color=cores)
    ax.axvline(0.10, color="#E8B93B", ls="--", lw=1)
    ax.axvline(0.25, color="#D64545", ls="--", lw=1)
    ax.set_xlabel("PSI (quanto a variável mudou)")
    ax.set_title("Monitoramento: variáveis que mais mudaram" + (" (cenário simulado)" if sufixo else ""))
    fig.tight_layout()
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PASTA_FIGURAS / f"monitoramento_psi{sufixo}.png", dpi=130)
    plt.close(fig)


def registrar_mlflow(resumo: dict, caminho_csv, sufixo: str) -> None:
    try:
        import mlflow
    except ImportError:
        print("MLflow não instalado: registro no MLflow ignorado (os arquivos em reports/ foram salvos).")
        return
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_MONITORAMENTO)
    with mlflow.start_run(run_name=f"monitoramento{sufixo or '_real'}"):
        mlflow.log_param("cenario", "simulado" if sufixo else "real")
        mlflow.log_param("modelo", resumo["modelo"])
        for chave in ("psi_maximo", "psi_score", "variaveis_em_alerta", "variaveis_em_atencao",
                      "roc_auc_atual", "roc_auc_referencia", "resultado_liquido_atual",
                      "resultado_liquido_referencia"):
            mlflow.log_metric(chave, float(resumo[chave]))
        mlflow.log_metric("alerta_geral", int(resumo["status_geral"] != "ok"))
        mlflow.log_artifact(str(caminho_csv))
    print(f"Monitoramento registrado no MLflow (experimento '{MLFLOW_MONITORAMENTO}').")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simular-drift", action="store_true")
    args = parser.parse_args()
    sufixo = "_simulado" if args.simular_drift else ""

    metricas = json.loads(METRICAS_JSON.read_text(encoding="utf-8"))
    p = ParametrosNegocio(**metricas["parametros_negocio"])
    modelo = joblib.load(MODELO_FINAL)
    df = carregar_tratado()
    treino, valid, teste = dividir_temporal(df)
    atual = simular_drift(teste) if args.simular_drift else teste

    # 1) PSI de cada variável
    linhas = []
    for col in FEATURES:
        psi = calcular_psi(treino[col], atual[col])
        linhas.append({
            "variavel": col, "psi": psi, "status": status_psi(psi),
            "tipo": "número" if col in COLS_NUM else "categoria",
        })
    tabela = pd.DataFrame(linhas).sort_values("psi", ascending=False).reset_index(drop=True)

    # 2) PSI do score e desempenho
    score_ref = pd.Series(modelo.predict_proba(valid[FEATURES])[:, 1])
    score_atual = pd.Series(modelo.predict_proba(atual[FEATURES])[:, 1])
    psi_score = calcular_psi(score_ref, score_atual)

    ref = metricas_negocio(valid[ALVO].values, score_ref.values, valid["valor_sinistro"].values, p)
    cur = metricas_negocio(atual[ALVO].values, score_atual.values, atual["valor_sinistro"].values, p)

    # 3) Alertas
    alertas = []
    n_alerta = int((tabela["status"] == "alerta").sum())
    n_atencao = int((tabela["status"] == "atenção").sum())
    if n_alerta:
        alertas.append(f"{n_alerta} variável(is) com PSI acima de 0,25: {', '.join(tabela.loc[tabela['status'] == 'alerta', 'variavel'].head(5))}")
    if psi_score >= 0.25:
        alertas.append(f"O score de risco mudou muito (PSI {psi_score:.2f})")
    if ref["roc_auc"] - cur["roc_auc"] > LIMITE_QUEDA_AUC:
        alertas.append(f"AUC caiu de {ref['roc_auc']:.3f} para {cur['roc_auc']:.3f}")
    if ref["resultado_liquido"] > 0 and cur["resultado_liquido"] < ref["resultado_liquido"] * (1 - LIMITE_QUEDA_RESULTADO):
        alertas.append("Resultado líquido caiu mais de 25% contra a validação")
    status_geral = "alerta" if alertas else ("atenção" if n_atencao or psi_score >= 0.10 else "ok")

    resumo = {
        "cenario": "simulado" if args.simular_drift else "real",
        "modelo": metricas["melhor_modelo"],
        "status_geral": status_geral,
        "alertas": alertas,
        "psi_maximo": float(tabela["psi"].max()),
        "psi_score": float(psi_score),
        "variaveis_em_alerta": n_alerta,
        "variaveis_em_atencao": n_atencao,
        "roc_auc_referencia": ref["roc_auc"],
        "roc_auc_atual": cur["roc_auc"],
        "pr_auc_referencia": ref["pr_auc"],
        "pr_auc_atual": cur["pr_auc"],
        "resultado_liquido_referencia": ref["resultado_liquido"],
        "resultado_liquido_atual": cur["resultado_liquido"],
        "taxa_fraude_referencia": float(valid[ALVO].mean()),
        "taxa_fraude_atual": float(atual[ALVO].mean()),
        "regras": {"psi_atencao": 0.10, "psi_alerta": 0.25, "queda_auc": LIMITE_QUEDA_AUC,
                   "queda_resultado": LIMITE_QUEDA_RESULTADO},
    }

    PASTA_REPORTS.mkdir(exist_ok=True)
    caminho_csv = PASTA_REPORTS / f"monitoramento{sufixo}.csv"
    tabela.to_csv(caminho_csv, index=False)
    (PASTA_REPORTS / f"monitoramento{sufixo}.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    grafico_psi(tabela, sufixo)

    print(f"Cenário: {resumo['cenario']} | status geral: {status_geral.upper()}")
    print(f"PSI máximo {resumo['psi_maximo']:.3f} | PSI do score {psi_score:.3f} | "
          f"alertas {n_alerta} | atenção {n_atencao}")
    print(f"AUC {ref['roc_auc']:.3f} -> {cur['roc_auc']:.3f} | "
          f"resultado líquido R$ {ref['resultado_liquido']:,.0f} -> R$ {cur['resultado_liquido']:,.0f}")
    for a in alertas:
        print(" ⚠", a)
    print("\nTop 6 variáveis por PSI:")
    print(tabela.head(6).round(3).to_string(index=False))
    registrar_mlflow(resumo, caminho_csv, sufixo)


if __name__ == "__main__":
    main()
