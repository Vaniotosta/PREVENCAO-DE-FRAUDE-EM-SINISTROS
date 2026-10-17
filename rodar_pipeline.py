"""
Roda TODAS as etapas do projeto na ordem certa.

    python rodar_pipeline.py                 # pipeline completo
    python rodar_pipeline.py --sem-mlflow    # pula a Etapa 6 (se o MLflow não estiver instalado)
    python rodar_pipeline.py --a-partir-de 4 # recomeça a partir da etapa 4

Depois, abra o app:
    streamlit run app/app.py
"""
import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent

ETAPAS = [
    (1, "Gerar dados sintéticos", ["src/01_gerar_dados.py"]),
    (2, "Pré-processamento (pandas)", ["src/02_preprocessamento.py"]),
    (3, "Análise exploratória (EDA)", ["src/03_eda.py"]),
    (4, "Modelagem (3 modelos)", ["src/04_modelagem.py"]),
    (5, "Priorização da fila", ["src/05_priorizacao.py"]),
    (6, "Empacotar no MLflow", ["src/06_mlflow_registro.py"]),
    (7, "Monitoramento (real)", ["src/07_monitoramento.py"]),
    (7, "Monitoramento (cenário simulado)", ["src/07_monitoramento.py", "--simular-drift"]),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sem-mlflow", action="store_true", help="pula a etapa 6")
    parser.add_argument("--a-partir-de", type=int, default=1, help="número da primeira etapa a rodar")
    args = parser.parse_args()

    tem_mlflow = importlib.util.find_spec("mlflow") is not None
    for numero, titulo, comando in ETAPAS:
        if numero < args.a_partir_de:
            continue
        if numero == 6 and (args.sem_mlflow or not tem_mlflow):
            motivo = "pedido do usuário" if args.sem_mlflow else "MLflow não instalado (pip install mlflow)"
            print(f"\n⏭️  Etapa 6 pulada: {motivo}")
            continue
        print(f"\n{'=' * 70}\n▶ Etapa {numero}: {titulo}\n{'=' * 70}")
        inicio = time.time()
        resultado = subprocess.run([sys.executable, *comando], cwd=RAIZ)
        if resultado.returncode != 0:
            print(f"\n❌ A etapa {numero} falhou. Corrija o erro acima e rode de novo com --a-partir-de {numero}")
            return resultado.returncode
        print(f"✔ Etapa {numero} concluída em {time.time() - inicio:.1f}s")

    print("\n✅ Pipeline completo. Abra o app com:  streamlit run app/app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
