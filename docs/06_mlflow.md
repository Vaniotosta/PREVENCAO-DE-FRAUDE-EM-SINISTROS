# Etapa 6 – Modelo empacotado no MLflow

Arquivo: `src/06_mlflow_registro.py` · Requer: `pip install mlflow`

> **Status de validação:** este script foi escrito e testado com um MLflow simulado, porque o ambiente de desenvolvimento não tinha internet para instalar o pacote. Na sua máquina, rode `python src/06_mlflow_registro.py` e confira a tela do MLflow. Se algum parâmetro de versão der erro, o script já tenta as variações de MLflow 2.x e 3.x.

## O que é registrado

| Item | Conteúdo |
|---|---|
| Experimento | `fraude_seguro_sinistros` (uma *run* por modelo) |
| Parâmetros | Modelo, nº de variáveis, linhas por conjunto, premissas de negócio e hiperparâmetros |
| Métricas | `validacao_*` e `teste_*`: AUC, PR-AUC, Brier, recall, precisão, lift, valor protegido, custo, resultado líquido, ROI |
| Artefatos | Gráficos, importâncias e `metricas.json` |
| Modelo | Pipeline completo (pré-processamento + modelo calibrado) com **assinatura** e exemplo de entrada |
| Registro | `modelo_fraude_seguro`, com o melhor modelo marcado com o alias `champion` |

O banco do MLflow é local: `mlflow.db` (SQLite) na raiz do projeto.

## Como usar

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # abre http://127.0.0.1:5000
```

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
modelo = mlflow.pyfunc.load_model("models:/modelo_fraude_seguro@champion")
probabilidades = modelo.predict(dados)   # devolve a probabilidade de fraude (predict_proba)
```

```bash
mlflow models serve -m "models:/modelo_fraude_seguro@champion" -p 5001 --env-manager local   # API REST
```

## Como o MLflow ajuda no monitoramento
- Cada retreino vira uma nova *run* comparável com as anteriores.
- O alias `champion` permite trocar de modelo sem mudar o código.
- A Etapa 7 grava no experimento `fraude_seguro_monitoramento` o PSI, o AUC e o resultado líquido de cada checagem.

O app Streamlit carrega `models/modelo_final.joblib` direto, sem depender do servidor do MLflow.
