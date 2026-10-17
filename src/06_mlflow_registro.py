"""
ETAPA 6 - Empacotar os modelos no MLflow.

O que este arquivo faz
----------------------
- Cria 1 "run" no MLflow para cada um dos 3 modelos, guardando:
  parâmetros, métricas (estatísticas e de negócio), gráficos e o modelo empacotado
  (com assinatura de entrada/saída e exemplo de entrada)
- Registra o melhor modelo no Model Registry como "modelo_fraude_seguro"
  e marca a versão com o apelido (alias) "champion"

Como rodar
----------
    python src/06_mlflow_registro.py

Para abrir a tela do MLflow:
    mlflow ui --backend-store-uri sqlite:///mlflow.db        (a partir da pasta do projeto)
    e acesse http://127.0.0.1:5000

Para servir o modelo como API (opcional):
    mlflow models serve -m "models:/modelo_fraude_seguro@champion" -p 5001 --env-manager local
"""
import json
import sys
import warnings

import joblib

from comum import (
    ALVO, FEATURES, METRICAS_JSON, MLFLOW_EXPERIMENTO, MLFLOW_URI, NOME_MODELO_REGISTRADO,
    PASTA_FIGURAS, PASTA_MODELOS, PASTA_REPORTS, ParametrosNegocio, carregar_tratado,
    dividir_temporal, metricas_negocio,
)

warnings.filterwarnings("ignore")

METRICAS_PARA_LOGAR = [
    "roc_auc", "pr_auc", "brier", "recall_na_capacidade", "precisao_na_capacidade",
    "lift_na_capacidade", "valor_protegido", "custo_investigacao", "resultado_liquido", "roi",
    "casos_investigados", "fraudes_capturadas",
]


def logar_modelo(mlflow, pipe, X, registrar: bool):
    """Salva o modelo no MLflow (compatível com MLflow 2.x e 3.x)."""
    import mlflow.sklearn
    from mlflow.models import infer_signature

    probabilidades = pipe.predict_proba(X.head(200))
    assinatura = infer_signature(X.head(200), probabilidades)
    base = dict(
        sk_model=pipe,
        signature=assinatura,
        input_example=X.head(3),
        registered_model_name=NOME_MODELO_REGISTRADO if registrar else None,
    )
    tentativas = [
        {"pyfunc_predict_fn": "predict_proba", "artifact_path": "modelo"},  # serve devolve probabilidades
        {"artifact_path": "modelo"},
        {"name": "modelo"},
    ]
    ultimo_erro = None
    for extra in tentativas:
        try:
            return mlflow.sklearn.log_model(**base, **extra)
        except TypeError as erro:  # versão do MLflow sem esse argumento
            ultimo_erro = erro
    raise ultimo_erro


def parametros_do_estimador(pipe) -> dict:
    calibrador = pipe.named_steps["modelo"]
    estimador = calibrador.estimator
    simples = {k: v for k, v in estimador.get_params().items() if isinstance(v, (int, float, str, bool, type(None)))}
    return {f"modelo__{k}": v for k, v in simples.items()}


def main() -> None:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        sys.exit("MLflow não está instalado. Rode:  pip install mlflow   e tente de novo.")

    if not METRICAS_JSON.exists():
        sys.exit("Rode antes: python src/04_modelagem.py")

    meta = json.loads(METRICAS_JSON.read_text(encoding="utf-8"))
    p = ParametrosNegocio(**meta["parametros_negocio"])
    melhor = meta["melhor_modelo"]
    df = carregar_tratado()
    treino, valid, teste = dividir_temporal(df)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENTO)
    cliente = MlflowClient()

    figuras = sorted(PASTA_FIGURAS.glob("modelo_*.png")) + sorted(PASTA_FIGURAS.glob("prioridade_*.png"))
    runs = {}
    for arquivo in sorted(PASTA_MODELOS.glob("*.joblib")):
        nome = arquivo.stem
        if nome == "modelo_final":
            continue
        pipe = joblib.load(arquivo)
        eh_melhor = nome == melhor

        with mlflow.start_run(run_name=nome) as run:
            mlflow.set_tags({"projeto": "prevencao_fraude_seguro", "etapa": "modelagem",
                             "melhor_modelo": str(eh_melhor)})
            mlflow.log_params({
                "modelo": nome,
                "n_variaveis": len(FEATURES),
                "linhas_treino": len(treino),
                "linhas_validacao": len(valid),
                "linhas_teste": len(teste),
                **{f"negocio__{k}": v for k, v in p.como_dict().items()},
                **parametros_do_estimador(pipe),
            })
            for conjunto, dados in (("validacao", valid), ("teste", teste)):
                prob = pipe.predict_proba(dados[FEATURES])[:, 1]
                m = metricas_negocio(dados[ALVO].values, prob, dados["valor_sinistro"].values, p)
                mlflow.log_metrics({f"{conjunto}_{k}": float(m[k]) for k in METRICAS_PARA_LOGAR})

            for figura in figuras:
                mlflow.log_artifact(str(figura), artifact_path="graficos")
            for extra in (PASTA_REPORTS / "importancia_permutacao.csv", METRICAS_JSON):
                if extra.exists():
                    mlflow.log_artifact(str(extra), artifact_path="documentacao")

            info = logar_modelo(mlflow, pipe, teste[FEATURES], registrar=eh_melhor)
            runs[nome] = run.info.run_id
            print(f"Run criada: {nome:<22} id={run.info.run_id[:8]}  {'<- melhor, registrado' if eh_melhor else ''}")

    # Alias "champion" na versão mais recente do modelo registrado
    versoes = cliente.search_model_versions(f"name='{NOME_MODELO_REGISTRADO}'")
    ultima = max(versoes, key=lambda v: int(v.version))
    cliente.set_registered_model_alias(NOME_MODELO_REGISTRADO, "champion", ultima.version)
    cliente.update_model_version(
        NOME_MODELO_REGISTRADO, ultima.version,
        description=f"Modelo {melhor} treinado com sinistros até {meta['divisao']['treino']['fim']}.",
    )
    print(f"\nModelo registrado: {NOME_MODELO_REGISTRADO} versão {ultima.version} (alias: champion)")
    print(f"Carregar em Python:  mlflow.sklearn.load_model('models:/{NOME_MODELO_REGISTRADO}@champion')")
    print("Abrir a interface:   mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
