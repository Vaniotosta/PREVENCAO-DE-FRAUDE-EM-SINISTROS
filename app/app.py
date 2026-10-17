"""
ETAPA 8 - App Streamlit: Prevenção de Fraude em Sinistros.

Como rodar (a partir da pasta do projeto):
    streamlit run app/app.py

O app NÃO treina nada: ele lê os arquivos gerados pelas etapas 1 a 7.
Se algum arquivo faltar, ele avisa e diz qual comando rodar.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from comum import (  # noqa: E402
    ALVO, COL_DATA, COL_ID, FEATURES, METRICAS_JSON, MODELO_FINAL, MOTIVOS, PASTA_FIGURAS,
    PASTA_REPORTS, PREDICOES, ParametrosNegocio, carregar_tratado, comparar_estrategias,
    construir_linha, curva_capacidade, dividir_temporal, distribuir_analistas, explicar_lote,
    metricas_negocio, rotular_prioridade, selecionar_casos, valor_esperado_liquido, valores_tipicos,
)

st.set_page_config(page_title="Prevenção de Fraude em Sinistros", page_icon="🛡️", layout="wide")

ESTADO_CIVIL = ["Solteiro", "Casado", "Divorciado", "Viúvo"]
COBERTURA = ["Compreensiva", "Roubo e Furto", "Colisão", "Básica"]
CANAL = ["Corretor", "Online", "Agência", "Telefone"]
TIPO_SINISTRO = ["Colisão", "Roubo", "Furto", "Incêndio", "Danos naturais"]
UF = ["SP", "RJ", "MG", "BA", "PR", "RS", "PE", "SC"]
DIAS_SEMANA = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
NOMES_MODELOS = {
    "regressao_logistica": "Regressão Logística",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


# --------------------------------------------------------------------------
# Carregamento (com cache)
# --------------------------------------------------------------------------
def arquivos_faltando() -> list:
    exigidos = {
        MODELO_FINAL: "python src/04_modelagem.py",
        METRICAS_JSON: "python src/04_modelagem.py",
        PREDICOES: "python src/04_modelagem.py",
        MOTIVOS: "python src/05_priorizacao.py",
    }
    return [(str(a.relative_to(RAIZ)), cmd) for a, cmd in exigidos.items() if not a.exists()]


@st.cache_resource
def carregar_modelo():
    return joblib.load(MODELO_FINAL)


@st.cache_data
def carregar_predicoes() -> pd.DataFrame:
    pred = pd.read_csv(PREDICOES, parse_dates=[COL_DATA])
    motivos = pd.read_csv(MOTIVOS)
    return pred.merge(motivos, on=COL_ID, how="left")


@st.cache_data
def carregar_metricas() -> dict:
    return json.loads(METRICAS_JSON.read_text(encoding="utf-8"))


@st.cache_data
def carregar_referencia():
    treino, _, _ = dividir_temporal(carregar_tratado())
    return valores_tipicos(treino), treino[ALVO].mean()


def mostrar_figura(nome: str, legenda: str = "") -> None:
    caminho = PASTA_FIGURAS / nome
    if caminho.exists():
        st.image(str(caminho), caption=legenda)
    else:
        st.info(f"Gráfico {nome} ainda não foi gerado.")


def moeda(valor: float) -> str:
    return f"R$ {valor:,.0f}".replace(",", ".")


# --------------------------------------------------------------------------
# Verificação inicial
# --------------------------------------------------------------------------
st.title("🛡️ Prevenção de Fraude em Sinistros")
st.caption("Priorize as investigações pelo valor que elas devem proteger, já descontando o custo de investigar.")

faltando = arquivos_faltando()
if faltando:
    st.error("Faltam arquivos gerados pelas etapas do projeto.")
    for arquivo, comando in faltando:
        st.write(f"• `{arquivo}` → rode `{comando}`")
    st.info("Atalho: `python rodar_pipeline.py` roda todas as etapas na ordem certa.")
    st.stop()

modelo = carregar_modelo()
pred = carregar_predicoes()
metricas = carregar_metricas()
tipicos, taxa_base_treino = carregar_referencia()
melhor = metricas["melhor_modelo"]
padrao = ParametrosNegocio(**metricas["parametros_negocio"])

# --------------------------------------------------------------------------
# Barra lateral: premissas de negócio
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Premissas de negócio")
custo = st.sidebar.number_input("Custo por investigação (R$)", min_value=100.0, max_value=20000.0,
                                value=float(padrao.custo_investigacao), step=100.0)
taxa = st.sidebar.slider("% do sinistro evitado quando a fraude é confirmada", 10, 100,
                         int(padrao.taxa_prevencao * 100), step=5) / 100
capacidade = st.sidebar.slider("Capacidade da equipe (% dos sinistros investigados)", 1, 40,
                               int(padrao.capacidade_pct * 100), step=1) / 100
n_analistas = st.sidebar.slider("Número de analistas", 1, 20, 5)
st.sidebar.caption(f"Modelo em uso: **{NOMES_MODELOS.get(melhor, melhor)}**")
p = ParametrosNegocio(custo_investigacao=custo, taxa_prevencao=taxa, capacidade_pct=capacidade)

aba_geral, aba_fila, aba_caso, aba_modelos, aba_monit, aba_sobre = st.tabs(
    ["📊 Visão geral", "📋 Fila de investigação", "🔎 Analisar um sinistro",
     "🧠 Modelos e explicabilidade", "📈 Monitoramento", "ℹ️ Sobre"]
)

y = pred[ALVO].values
prob = pred["prob_modelo"].values
valor = pred["valor_sinistro"].values

# --------------------------------------------------------------------------
# Aba 1: visão geral
# --------------------------------------------------------------------------
with aba_geral:
    st.subheader("Resultado esperado com os dados históricos de teste")
    st.caption("Simulação: usamos sinistros já encerrados para medir o que teria acontecido seguindo a fila do modelo.")
    m = metricas_negocio(y, prob, valor, p)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sinistros analisados", f"{len(pred):,}".replace(",", "."))
    c2.metric("Investigados (capacidade)", f"{m['casos_investigados']:,}".replace(",", "."))
    c3.metric("Fraudes encontradas", f"{m['fraudes_capturadas']} de {m['fraudes_total']}")
    c4.metric("Precisão da fila", f"{m['precisao_na_capacidade']:.0%}", f"{m['lift_na_capacidade']:.1f}x melhor que sortear")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Valor protegido", moeda(m["valor_protegido"]))
    c6.metric("Custo das investigações", moeda(m["custo_investigacao"]))
    c7.metric("Resultado líquido", moeda(m["resultado_liquido"]))
    c8.metric("Retorno (ROI)", f"{m['roi']:.1f}x")

    st.divider()
    esq, dir_ = st.columns(2)
    with esq:
        st.subheader("Modelo x alternativas")
        tabela = comparar_estrategias(y, prob, valor, p)
        st.bar_chart(tabela.set_index("estrategia")["resultado_liquido"])
        exibir = tabela.copy()
        for col in ("valor_protegido", "custo", "resultado_liquido"):
            exibir[col] = exibir[col].map(moeda)
        st.dataframe(exibir.rename(columns={
            "estrategia": "Estratégia", "casos_investigados": "Casos", "fraudes_capturadas": "Fraudes",
            "valor_protegido": "Valor protegido", "custo": "Custo", "resultado_liquido": "Resultado líquido"}),
            hide_index=True)
    with dir_:
        st.subheader("Vale ampliar a equipe?")
        curva = curva_capacidade(y, prob, valor, p)
        st.line_chart(curva.set_index("capacidade_pct")["resultado_liquido"])
        ponto = curva.loc[curva["resultado_liquido"].idxmax()]
        st.write(f"Com estas premissas, o melhor resultado aparece investigando **{ponto['capacidade_pct']:.0%}** "
                 f"dos sinistros ({moeda(ponto['resultado_liquido'])}). "
                 "Depois disso, cada investigação extra custa mais do que protege.")

# --------------------------------------------------------------------------
# Aba 2: fila
# --------------------------------------------------------------------------
with aba_fila:
    st.subheader("Fila priorizada de investigação")
    pred_fila = pred.copy()
    pred_fila["valor_esperado_liquido"] = valor_esperado_liquido(prob, valor, p)
    pred_fila["valor_protegido_esperado"] = prob * valor * taxa
    fila = pred_fila.loc[selecionar_casos(prob, valor, p)].sort_values("valor_esperado_liquido", ascending=False).copy()
    fila.insert(0, "posicao", np.arange(1, len(fila) + 1))
    fila["prioridade"] = rotular_prioridade(fila["posicao"], len(fila))
    fila["analista"] = distribuir_analistas(fila["posicao"], n_analistas)

    f1, f2 = st.columns(2)
    filtro_analista = f1.selectbox("Analista", ["Todos"] + sorted(fila["analista"].unique(), key=lambda s: int(s.split()[-1])))
    filtro_prioridade = f2.selectbox("Prioridade", ["Todas", "Crítica", "Alta", "Média"])
    visao = fila
    if filtro_analista != "Todos":
        visao = visao[visao["analista"] == filtro_analista]
    if filtro_prioridade != "Todas":
        visao = visao[visao["prioridade"] == filtro_prioridade]

    mostrar = visao[["posicao", "prioridade", "analista", COL_ID, "prob_modelo", "valor_sinistro",
                     "valor_esperado_liquido", "motivo_principal", "motivo_secundario", "tipo_sinistro", "uf"]].copy()
    mostrar["prob_modelo"] = (mostrar["prob_modelo"] * 100).round(1)
    mostrar["valor_sinistro"] = mostrar["valor_sinistro"].round(0)
    mostrar["valor_esperado_liquido"] = mostrar["valor_esperado_liquido"].round(0)
    mostrar = mostrar.rename(columns={
        "posicao": "Posição", "prioridade": "Prioridade", "analista": "Analista", COL_ID: "Sinistro",
        "prob_modelo": "Prob. fraude (%)", "valor_sinistro": "Valor do sinistro (R$)",
        "valor_esperado_liquido": "Valor esperado líquido (R$)", "motivo_principal": "Motivo principal",
        "motivo_secundario": "Motivo secundário", "tipo_sinistro": "Tipo", "uf": "UF"})
    st.write(f"{len(visao)} casos exibidos de {len(fila)} na fila.")
    st.dataframe(mostrar, hide_index=True, height=480)
    st.download_button("⬇️ Baixar fila em CSV", mostrar.to_csv(index=False).encode("utf-8"),
                       file_name="fila_priorizada.csv", mime="text/csv")

    st.subheader("Carga por analista")
    carga = fila.groupby("analista").agg(casos=(COL_ID, "count"), valor_esperado=("valor_esperado_liquido", "sum"))
    st.bar_chart(carga["valor_esperado"])

# --------------------------------------------------------------------------
# Aba 3: analisar um sinistro
# --------------------------------------------------------------------------
with aba_caso:
    st.subheader("Simular um sinistro novo")
    st.caption("Preencha os dados e veja a probabilidade de fraude, a decisão e o porquê.")
    with st.form("form_sinistro"):
        st.markdown("**Segurado**")
        a1, a2, a3, a4 = st.columns(4)
        idade = a1.number_input("Idade", 18, 85, 42)
        tempo = a2.number_input("Anos como cliente", 0.0, 30.0, 4.0, step=0.5)
        score = a3.number_input("Score de crédito", 250, 950, 650)
        renda = a4.number_input("Renda mensal (R$)", 1500.0, 60000.0, 5000.0, step=100.0)
        estado_civil = a1.selectbox("Estado civil", ESTADO_CIVIL)

        st.markdown("**Apólice**")
        b1, b2, b3, b4 = st.columns(4)
        cobertura = b1.selectbox("Cobertura", COBERTURA)
        valor_segurado = b2.number_input("Valor segurado (R$)", 15000.0, 400000.0, 60000.0, step=1000.0)
        franquia = b3.number_input("Franquia (R$)", 0.0, 40000.0, 3600.0, step=100.0)
        premio = b4.number_input("Prêmio anual (R$)", 500.0, 30000.0, 3000.0, step=100.0)
        canal = b1.selectbox("Canal de venda", CANAL)
        alteracoes = b2.number_input("Alterações na apólice (90 dias)", 0, 10, 0)

        st.markdown("**Histórico**")
        h1, h2 = st.columns(2)
        sinistros_12m = h1.number_input("Sinistros nos últimos 12 meses", 0, 10, 0)
        sinistros_total = h2.number_input("Sinistros no total", 0, 20, 1)

        st.markdown("**Acidente**")
        c1, c2, c3, c4 = st.columns(4)
        tipo = c1.selectbox("Tipo de sinistro", TIPO_SINISTRO)
        valor_sinistro = c2.number_input("Valor do sinistro (R$)", 500.0, 400000.0, 20000.0, step=500.0)
        dias = c3.number_input("Dias entre contratar e sinistrar", 0, 1500, 150)
        atraso = c4.number_input("Dias para avisar a seguradora", 0, 90, 1)
        hora = c1.slider("Hora do sinistro", 0, 23, 14)
        dia_semana = c2.selectbox("Dia da semana", DIAS_SEMANA)
        uf = c3.selectbox("Estado", UF)
        bo = c4.selectbox("Boletim de ocorrência", ["Sim", "Não"])
        testemunhas = c1.number_input("Testemunhas", 0, 10, 1)
        vitima = c2.selectbox("Houve vítima?", ["Não", "Sim"])
        oficina = c3.selectbox("Oficina credenciada?", ["Sim", "Não"])
        enviado = st.form_submit_button("Calcular risco")

    if enviado:
        entrada = {
            "idade": idade, "tempo_cliente_anos": tempo, "score_credito": score, "renda_mensal": renda,
            "estado_civil": estado_civil, "tipo_cobertura": cobertura, "valor_segurado": valor_segurado,
            "franquia": franquia, "premio_anual": premio, "canal_venda": canal,
            "alteracoes_apolice_90d": alteracoes, "qtd_sinistros_12m": sinistros_12m,
            "qtd_sinistros_total": max(sinistros_total, sinistros_12m),
            "dias_inicio_ate_sinistro": dias, "atraso_aviso_dias": atraso,
            "dia_semana": DIAS_SEMANA.index(dia_semana), "tipo_sinistro": tipo, "hora_sinistro": hora,
            "uf": uf, "boletim_ocorrencia": int(bo == "Sim"), "qtd_testemunhas": testemunhas,
            "houve_vitima": int(vitima == "Sim"), "oficina_credenciada": int(oficina == "Sim"),
            "valor_sinistro": valor_sinistro,
        }
        linha = construir_linha(entrada)
        prob_caso = float(modelo.predict_proba(linha[FEATURES])[0, 1])
        esperado = float(valor_esperado_liquido(prob_caso, valor_sinistro, p))
        percentil = float((pred["prob_modelo"] < prob_caso).mean())

        r1, r2, r3 = st.columns(3)
        r1.metric("Probabilidade de fraude", f"{prob_caso:.1%}", f"base histórica: {taxa_base_treino:.1%}", delta_color="off")
        r2.metric("Valor esperado líquido", moeda(esperado))
        r3.metric("Mais arriscado que", f"{percentil:.0%} dos sinistros")
        if esperado > 0:
            st.warning("**Recomendação: investigar.** O valor que se espera proteger é maior que o custo da investigação.")
        else:
            st.success("**Recomendação: liberar o fluxo normal.** Investigar custaria mais do que se espera proteger.")

        contribuicoes = explicar_lote(modelo, linha, tipicos).iloc[0].sort_values()
        st.markdown("**Por que o modelo chegou nesse resultado?**")
        st.caption("Barras positivas aumentam o risco; negativas diminuem. Medido trocando cada grupo por um valor típico.")
        st.bar_chart((contribuicoes * 100).rename("efeito na probabilidade (pontos percentuais)"), horizontal=True)
        alertas = contribuicoes[contribuicoes > 0.005].sort_values(ascending=False).head(3)
        if len(alertas):
            st.write("Principais sinais de alerta: " + "; ".join(f"{k} (+{v * 100:.1f} p.p.)" for k, v in alertas.items()) + ".")

# --------------------------------------------------------------------------
# Aba 4: modelos e explicabilidade
# --------------------------------------------------------------------------
with aba_modelos:
    st.subheader("Comparação dos 3 modelos")
    st.caption(f"Escolha do modelo: maior resultado líquido na validação → **{NOMES_MODELOS.get(melhor, melhor)}**. "
               "O teste é usado só para a nota final.")
    res = pd.DataFrame(metricas["resultados"])
    conjunto = st.radio("Base de avaliação", ["teste", "validacao"], horizontal=True)
    tabela_m = res[res["conjunto"] == conjunto].copy()
    tabela_m["modelo"] = tabela_m["modelo"].map(NOMES_MODELOS)
    tabela_m = tabela_m[["modelo", "roc_auc", "pr_auc", "recall_na_capacidade", "precisao_na_capacidade",
                         "lift_na_capacidade", "valor_protegido", "custo_investigacao", "resultado_liquido", "roi"]]
    st.dataframe(tabela_m.rename(columns={
        "modelo": "Modelo", "roc_auc": "AUC", "pr_auc": "PR-AUC", "recall_na_capacidade": "Recall na capacidade",
        "precisao_na_capacidade": "Precisão na capacidade", "lift_na_capacidade": "Lift",
        "valor_protegido": "Valor protegido", "custo_investigacao": "Custo", "resultado_liquido": "Resultado líquido",
        "roi": "ROI"}).round(3), hide_index=True)
    g1, g2 = st.columns(2)
    with g1:
        mostrar_figura("modelo_02_resultado_liquido.png")
        mostrar_figura("modelo_04_ganho_acumulado.png")
    with g2:
        mostrar_figura("modelo_01_roc_pr.png")
        mostrar_figura("modelo_03_calibracao.png")
    st.subheader("Explicabilidade global")
    st.write("Importância por permutação: quanto o desempenho piora quando embaralhamos uma variável. "
             "Quanto maior a barra, mais o modelo depende dela.")
    mostrar_figura("modelo_05_importancia.png")

# --------------------------------------------------------------------------
# Aba 5: monitoramento
# --------------------------------------------------------------------------
with aba_monit:
    st.subheader("Saúde do modelo")
    cenario = st.radio("Cenário", ["Dados reais (teste)", "Cenário simulado de mudança"], horizontal=True)
    sufixo = "_simulado" if cenario.startswith("Cenário") else ""
    arq_json = PASTA_REPORTS / f"monitoramento{sufixo}.json"
    arq_csv = PASTA_REPORTS / f"monitoramento{sufixo}.csv"
    if not arq_json.exists():
        comando = "python src/07_monitoramento.py" + (" --simular-drift" if sufixo else "")
        st.info(f"Ainda não há relatório para este cenário. Rode `{comando}`.")
    else:
        mon = json.loads(arq_json.read_text(encoding="utf-8"))
        status = mon["status_geral"]
        if status == "ok":
            st.success("Status geral: **OK** — nenhuma mudança relevante.")
        elif status == "atenção":
            st.warning("Status geral: **ATENÇÃO** — acompanhe de perto.")
        else:
            st.error("Status geral: **ALERTA** — revise os dados e considere retreinar o modelo.")
        for alerta in mon["alertas"]:
            st.write("⚠️ " + alerta)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("PSI máximo", f"{mon['psi_maximo']:.2f}")
        k2.metric("PSI do score", f"{mon['psi_score']:.2f}")
        k3.metric("AUC", f"{mon['roc_auc_atual']:.3f}", f"{mon['roc_auc_atual'] - mon['roc_auc_referencia']:+.3f}")
        k4.metric("Resultado líquido", moeda(mon["resultado_liquido_atual"]),
                  moeda(mon["resultado_liquido_atual"] - mon["resultado_liquido_referencia"]))
        st.caption("PSI abaixo de 0,10 = estável | 0,10 a 0,25 = atenção | acima de 0,25 = alerta.")
        m1, m2 = st.columns([1, 1])
        with m1:
            mostrar_figura(f"monitoramento_psi{sufixo}.png")
        with m2:
            if arq_csv.exists():
                st.dataframe(pd.read_csv(arq_csv).head(15).round(3), hide_index=True)

# --------------------------------------------------------------------------
# Aba 6: sobre
# --------------------------------------------------------------------------
with aba_sobre:
    st.subheader("Sobre o projeto")
    st.markdown(
        """
**Problema:** investigar todos os sinistros é caro e impossível; investigar poucos deixa a fraude passar.

**Solução:** um modelo estima a probabilidade de fraude de cada sinistro. Com ela, calculamos o
*valor esperado líquido* de investigar:

`probabilidade de fraude × valor do sinistro × % evitado − custo da investigação`

Os casos são ordenados do maior para o menor valor, respeitando a capacidade da equipe
e divididos entre os analistas.

**Dados:** 100% fictícios, criados para estudo. Nenhuma informação pessoal real.

**Limites:** as premissas de custo e de % evitado são ilustrativas. A fila é um apoio à decisão:
a confirmação da fraude é sempre do analista. Veja o `docs/MODEL_CARD.md` para detalhes.
"""
    )
