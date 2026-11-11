"""Etapa 4: painel com os motores em risco (Streamlit).

Dados: NASA C-MAPSS, subconjunto FD001 (simulados e públicos, não são dados da GE).

Como rodar:
    .venv/bin/streamlit run src/04_painel.py

O painel lê do banco:
  - previsoes_teste  → RUL previsto pelo modelo da etapa 3 para cada motor de teste
  - leituras_teste   → histórico dos sensores de cada motor de teste
  - leituras_treino  → usado só para calcular os valores de referência
                       (como um sensor fica num motor saudável e perto da falha)

Streamlit funciona assim: o script inteiro roda de cima a baixo toda vez que o usuário
mexe em algum controle (slider, caixa de seleção). Por isso a leitura do banco fica
num cache (@st.cache_data), para não reler o banco a cada clique.
"""
import sqlite3
import subprocess
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
BANCO = RAIZ / "dados" / "motores.db"

# Cores de status: vermelho, laranja e verde. Elas sempre aparecem junto com
# um ícone e um nome ("🔴 Crítico"), para não depender só da cor (daltonismo).
STATUS = {
    "🔴 Crítico": "#d03b3b",
    "🟠 Atenção": "#ec835a",
    "🟢 Saudável": "#0ca30c",
}
AZUL = "#2a78d6"
CINZA = "#b5b3ad"

# Sensores mostrados no detalhe de cada motor: os mais importantes na etapa 3
SENSORES_DETALHE = {
    "ps30_pressao_estatica_hpc": "Pressão estática na saída do HPC (ps30)",
    "t50_temp_saida_lpt": "Temperatura na saída da LPT (t50)",
    "nc_rotacao_nucleo": "Rotação do núcleo (nc)",
    "phi_razao_combustivel_ps30": "Razão combustível / ps30 (phi)",
}


def dados_prontos() -> bool:
    """O painel precisa da tabela previsoes_teste, criada pela etapa 3."""
    if not BANCO.exists():
        return False
    with sqlite3.connect(BANCO) as con:
        existe = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'previsoes_teste'"
        ).fetchone()
    return existe is not None


def preparar_dados() -> None:
    """Roda as etapas 1, 2 e 3 em sequência, caso o banco ainda não exista.

    Isso acontece num computador novo (ou num servidor na nuvem): o banco e o modelo
    ficam fora do git, então o painel os gera a partir dos arquivos brutos da NASA.
    sys.executable é o mesmo Python que está rodando o painel (o da .venv), e
    check=True faz o painel parar com erro se alguma etapa falhar.
    """
    for script in ["01_carregar_banco.py", "02_analise_exploratoria.py", "03_modelo_rul.py"]:
        subprocess.run([sys.executable, str(RAIZ / "src" / script)], check=True, capture_output=True)


@st.cache_data
def carregar() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(BANCO) as con:
        previsoes = pd.read_sql("SELECT * FROM previsoes_teste", con)
        leituras = pd.read_sql("SELECT * FROM leituras_teste", con)
        # Referências calculadas no próprio SQL: média de cada sensor quando o motor
        # estava saudável (RUL >= 125) e quando estava perto da falha (RUL <= 10).
        colunas = ", ".join(f"AVG({s}) AS {s}" for s in SENSORES_DETALHE)
        referencias = pd.read_sql(
            f"""
            SELECT 'Média de motor saudável' AS referencia, {colunas}
            FROM leituras_treino WHERE rul >= 125
            UNION ALL
            SELECT 'Média perto da falha' AS referencia, {colunas}
            FROM leituras_treino WHERE rul <= 10
            """,
            con,
        )
    return previsoes, leituras, referencias


def classificar(rul: float, critico: int, atencao: int) -> str:
    if rul <= critico:
        return "🔴 Crítico"
    if rul <= atencao:
        return "🟠 Atenção"
    return "🟢 Saudável"


def mostrar_grafico(grafico: alt.Chart) -> None:
    """Mostra o gráfico ocupando toda a largura da coluna.

    O painel roda em duas versões do Streamlit: a 1.50 da .venv (Python 3.9) e a
    versão mais nova que o stlite usa no navegador (site). A nova usa width="stretch";
    a antiga não conhece esse parâmetro e dá TypeError, então caímos na forma antiga.
    """
    try:
        st.altair_chart(grafico, width="stretch")
    except TypeError:
        st.altair_chart(grafico, use_container_width=True)


def escala_status() -> alt.Scale:
    return alt.Scale(domain=list(STATUS), range=list(STATUS.values()))


def grafico_ranking(previsoes: pd.DataFrame, quantos: int = 20) -> alt.Chart:
    """Barras horizontais com os motores de menor vida prevista, coloridos pelo status."""
    top = previsoes.nsmallest(quantos, "rul_previsto").copy()
    top["nome"] = "Motor " + top["motor"].astype(str)
    return (
        alt.Chart(top)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("nome:N", sort=alt.SortField("rul_previsto"), title=None,
                    scale=alt.Scale(paddingInner=0.3)),  # 30% de espaço entre as barras
            x=alt.X("rul_previsto:Q", title="vida útil restante prevista (ciclos)"),
            color=alt.Color("status:N", scale=escala_status(), legend=alt.Legend(title=None, orient="top")),
            tooltip=[
                alt.Tooltip("nome:N", title="Motor"),
                alt.Tooltip("rul_previsto:Q", title="RUL previsto", format=".0f"),
                alt.Tooltip("ultimo_ciclo:Q", title="Ciclos já voados"),
                alt.Tooltip("status:N", title="Status"),
            ],
        )
        .properties(height=alt.Step(20))  # cada motor ocupa 20 px de altura
    )


def grafico_sensor(historico: pd.DataFrame, referencias: pd.DataFrame, sensor: str) -> alt.Chart:
    """Histórico de um sensor do motor escolhido, com as duas linhas de referência.

    Cinza: leitura bruta de cada ciclo (tem muito ruído).
    Azul: média móvel dos últimos 10 ciclos, que mostra a tendência.
    Tracejadas: onde o sensor costuma ficar num motor saudável e perto da falha.
    Se a linha azul está indo em direção à tracejada vermelha, o motor está degradando.
    """
    dados = historico[["ciclo", sensor]].copy()
    dados["media_movel"] = dados[sensor].rolling(10, min_periods=1).mean()
    base = alt.Chart(dados).encode(x=alt.X("ciclo:Q", title="ciclo"))
    bruto = base.mark_line(color=CINZA, strokeWidth=1).encode(
        y=alt.Y(f"{sensor}:Q", title=None, scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("ciclo:Q"), alt.Tooltip(f"{sensor}:Q", title="leitura", format=".2f")],
    )
    tendencia = base.mark_line(color=AZUL, strokeWidth=2).encode(y=f"media_movel:Q")
    linhas_ref = (
        alt.Chart(referencias[["referencia", sensor]])
        .mark_rule(strokeDash=[5, 4], strokeWidth=1.5)
        .encode(
            y=f"{sensor}:Q",
            color=alt.Color(
                "referencia:N",
                scale=alt.Scale(
                    domain=["Média de motor saudável", "Média perto da falha"],
                    range=[STATUS["🟢 Saudável"], STATUS["🔴 Crítico"]],
                ),
                legend=None,  # explicado na legenda de texto embaixo dos 4 gráficos
            ),
            tooltip=[alt.Tooltip("referencia:N", title=" "), alt.Tooltip(f"{sensor}:Q", title="valor", format=".2f")],
        )
    )
    return (bruto + tendencia + linhas_ref).properties(height=300, title=SENSORES_DETALHE[sensor])  # altura total: título, eixos e linhas


def main() -> None:
    st.set_page_config(page_title="Monitor de motores turbofan", page_icon="✈️", layout="wide")
    if not dados_prontos():
        with st.spinner("Primeira execução: criando o banco e treinando o modelo (cerca de 30 segundos)..."):
            preparar_dados()
    previsoes, leituras, referencias = carregar()

    # --- Barra lateral: limites de risco ajustáveis ---
    st.sidebar.header("Limites de risco")
    st.sidebar.caption(
        "O modelo erra em média ~14 ciclos e em metade dos casos é otimista. "
        "Por isso os limites têm folga: melhor revisar cedo do que tarde."
    )
    critico = st.sidebar.slider("Crítico: até quantos ciclos de vida restante", 5, 60, 30, step=5)
    atencao = st.sidebar.slider("Atenção: até quantos ciclos", critico + 5, 125, max(60, critico + 5), step=5)

    previsoes["status"] = previsoes["rul_previsto"].apply(classificar, args=(critico, atencao))

    # --- Cabeçalho ---
    st.title("✈️ Monitor de saúde de motores turbofan")
    st.caption(
        "Dados **simulados e públicos** da NASA (C-MAPSS FD001, Saxena e Goebel, 2008). "
        "Não são dados reais da GE nem de nenhuma companhia aérea."
    )

    # --- Resumo da frota ---
    contagem = previsoes["status"].value_counts()
    colunas = st.columns(3)
    for coluna, status in zip(colunas, STATUS):
        coluna.metric(status, f"{contagem.get(status, 0)} motores")

    # --- Ranking e tabela ---
    esquerda, direita = st.columns([1, 1])
    with esquerda:
        st.subheader("20 motores com menos vida prevista")
        mostrar_grafico(grafico_ranking(previsoes))
    with direita:
        st.subheader("Frota completa")
        tabela = previsoes.sort_values("rul_previsto")[["motor", "status", "rul_previsto", "ultimo_ciclo"]]
        st.dataframe(
            tabela,
            hide_index=True,
            height=440,
            width="stretch",
            column_config={
                "motor": st.column_config.NumberColumn("Motor", format="%d"),
                "status": "Status",
                "rul_previsto": st.column_config.ProgressColumn(
                    "RUL previsto (ciclos)", min_value=0, max_value=125, format="%.0f"
                ),
                "ultimo_ciclo": st.column_config.NumberColumn("Ciclos já voados", format="%d"),
            },
        )

    # --- Detalhe de um motor ---
    st.divider()
    st.subheader("Detalhe do motor")
    ordem = previsoes.sort_values("rul_previsto")["motor"].tolist()  # o mais crítico vem primeiro
    motor = st.selectbox("Escolha um motor (ordenados do mais crítico ao mais saudável)", ordem,
                         format_func=lambda m: f"Motor {m}")
    linha = previsoes.set_index("motor").loc[motor]

    c1, c2, c3 = st.columns(3)
    c1.metric("Status", linha["status"])
    c2.metric("Vida útil restante prevista", f"{linha['rul_previsto']:.0f} ciclos")
    c3.metric("Ciclos já voados", int(linha["ultimo_ciclo"]))

    historico = leituras[leituras["motor"] == motor].sort_values("ciclo")
    grade = st.columns(2)
    for i, sensor in enumerate(SENSORES_DETALHE):
        with grade[i % 2]:
            mostrar_grafico(grafico_sensor(historico, referencias, sensor))
    st.caption("Cinza: leitura de cada ciclo · Azul: média dos últimos 10 ciclos · "
               "Tracejada verde: média de motor saudável · Tracejada vermelha: média perto da falha "
               "(as duas calculadas nos motores de treino)")

    # --- Transparência: comparação com o RUL real ---
    with st.expander("Como o modelo se saiu neste motor?"):
        st.write(
            "Num motor de verdade não saberíamos o RUL real; aqui sabemos porque é o conjunto "
            "de teste da NASA, onde a resposta certa é publicada."
        )
        erro = linha["rul_previsto"] - linha["rul_real"]
        st.write(
            f"RUL real: **{linha['rul_real']:.0f} ciclos** · previsto: **{linha['rul_previsto']:.0f}** · "
            f"erro: **{erro:+.0f}** ({'otimista' if erro > 0 else 'conservador'})"
        )
        mae = (previsoes["rul_previsto"] - previsoes["rul_real"]).abs().mean()
        st.write(f"Erro médio do modelo nos 100 motores de teste: **{mae:.1f} ciclos**.")


main()
