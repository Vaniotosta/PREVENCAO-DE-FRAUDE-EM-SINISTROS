"""Faz os testes enxergarem a pasta src/ (rode: pytest -q a partir da pasta do projeto)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
