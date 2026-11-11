"""Etapa 2: análise exploratória — quais sensores mudam quando o motor degrada?

Dados: NASA C-MAPSS, subconjunto FD001 (simulados e públicos, não são dados da GE).

O script responde a três perguntas, usando só a tabela leituras_treino
(lá cada motor roda até falhar, então sabemos o RUL de cada leitura):

  1. Quais sensores são constantes e não dizem nada sobre o desgaste?
  2. Dos que variam, quais acompanham a degradação (sobem ou descem perto da falha)?
  3. Há sensores que repetem a mesma informação (redundantes)?

Saídas:
  - resumo no terminal
  - docs/figuras/01_sensores_ate_falha.png  → curva de cada sensor nos últimos ciclos
  - docs/figuras/02_correlacao_com_rul.png  → ranking dos sensores pela correlação com o RUL
  - dados/sensores_uteis.txt                → lista de sensores que a etapa 3 vai usar
"""
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera os PNGs sem abrir janela
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
BANCO = RAIZ / "dados" / "motores.db"
FIGURAS = RAIZ / "docs" / "figuras"

# Quantos ciclos antes da falha vamos olhar nos gráficos.
# O motor mais curto do treino durou 128 ciclos e a maioria passa de 200,
# então com 200 quase todos os motores aparecem em quase todos os pontos.
JANELA = 200

# Cores (azul e laranja: um par que continua distinguível para daltônicos)
AZUL = "#2a78d6"
LARANJA = "#eb6834"
CINZA = "#b5b3ad"
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
FUNDO = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": FUNDO,
    "axes.facecolor": FUNDO,
    "axes.edgecolor": CINZA,
    "axes.labelcolor": TEXTO_2,
    "xtick.color": TEXTO_2,
    "ytick.color": TEXTO_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,             # grade atrás das barras e linhas
    "axes.formatter.useoffset": False,  # mostra 2388.1 no eixo, e não "0.1 +2.388e3"
    "font.size": 9,
})


def carregar() -> pd.DataFrame:
    with sqlite3.connect(BANCO) as con:
        return pd.read_sql("SELECT * FROM leituras_treino", con)


def separar_sensores(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Pergunta 1: separa sensores constantes dos que variam.

    Critério: quantos valores diferentes o sensor registrou nas 20 mil leituras.
    Com 1 ou 2 valores, ele não tem como acompanhar um desgaste gradual.
    (Não dá para usar o desvio-padrão puro, porque cada sensor tem uma escala:
    a rotação do fan varia só 0,07 rpm em média e mesmo assim acompanha a falha.)
    """
    sensores = [c for c in df.columns if c not in ("motor", "ciclo", "rul") and not c.startswith("config")]
    valores_distintos = df[sensores].nunique()
    constantes = valores_distintos[valores_distintos <= 2].index.tolist()
    uteis = valores_distintos[valores_distintos > 2].index.tolist()
    return uteis, constantes


def correlacao_com_rul(df: pd.DataFrame, sensores: list[str]) -> pd.Series:
    """Pergunta 2: correlação de Pearson de cada sensor com o RUL.

    Vai de -1 a +1.
      - Negativa: quando o RUL cai (motor perto da falha), o sensor SOBE.
      - Positiva: quando o RUL cai, o sensor DESCE.
      - Perto de 0: o sensor não acompanha o desgaste de forma linear.
    """
    return df[sensores].corrwith(df["rul"]).sort_values(key=abs, ascending=False)


def pares_redundantes(df: pd.DataFrame, sensores: list[str], limite: float = 0.9) -> list[tuple]:
    """Pergunta 3: pares de sensores com correlação acima do limite entre si.

    Se dois sensores andam juntos quase sempre, eles carregam a mesma informação.
    """
    matriz = df[sensores].corr()
    pares = []
    for i, a in enumerate(sensores):
        for b in sensores[i + 1:]:  # só a metade de cima da matriz, sem repetir pares
            if abs(matriz.loc[a, b]) >= limite:
                pares.append((a, b, matriz.loc[a, b]))
    return pares


def grafico_sensores_ate_falha(df: pd.DataFrame, sensores: list[str], corr: pd.Series) -> Path:
    """Figura 1: um painel por sensor, com o eixo x em 'ciclos antes da falha'.

    Alinhar todos os motores pelo momento da falha (x = -RUL) permite comparar
    motores que viveram 128 ciclos com motores que viveram 362.
    Linhas cinzas: 5 motores individuais (mostram o ruído da medição).
    Linha azul: média de todos os motores em cada ponto (mostra a tendência).
    """
    recorte = df[df["rul"] <= JANELA].copy()
    recorte["antes_da_falha"] = -recorte["rul"]
    media = recorte.groupby("antes_da_falha")[sensores].mean()
    exemplos = recorte[recorte["motor"].isin([1, 2, 3, 4, 5])]

    ordem = corr.index.tolist()  # sensores mais ligados ao desgaste primeiro
    colunas = 4
    linhas = -(-len(ordem) // colunas)  # divisão arredondada para cima
    fig, eixos = plt.subplots(linhas, colunas, figsize=(14, 2.6 * linhas), sharex=True)

    for eixo, sensor in zip(eixos.flat, ordem):
        for _, motor in exemplos.groupby("motor"):
            eixo.plot(motor["antes_da_falha"], motor[sensor], color=CINZA, linewidth=0.7, alpha=0.8)
        eixo.plot(media.index, media[sensor], color=AZUL, linewidth=2)
        eixo.set_title(f"{sensor}\ncorrelação com RUL: {corr[sensor]:+.2f}", fontsize=8.5, color=TEXTO, loc="left")
        eixo.axvline(0, color=TEXTO_2, linewidth=0.8, linestyle=":")

    for eixo in list(eixos.flat)[len(ordem):]:
        eixo.set_visible(False)  # esconde painéis sobrando na grade
    for eixo in eixos[-1]:
        eixo.set_xlabel("ciclos antes da falha")

    fig.suptitle(
        "Como cada sensor se comporta nos últimos 200 ciclos antes da falha\n"
        "azul = média dos 100 motores · cinza = motores 1 a 5 · dados simulados NASA C-MAPSS FD001",
        x=0.01, ha="left", fontsize=11, color=TEXTO,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    caminho = FIGURAS / "01_sensores_ate_falha.png"
    fig.savefig(caminho, dpi=130)
    plt.close(fig)
    return caminho


def grafico_correlacao(corr: pd.Series) -> Path:
    """Figura 2: barras horizontais com a correlação de cada sensor com o RUL.

    Mostramos o valor absoluto (o tamanho da barra é a força da relação)
    e usamos a cor para a direção: sobe ou desce perto da falha.
    """
    dados = corr.iloc[::-1]  # inverte para o mais forte ficar no topo
    cores = [LARANJA if v < 0 else AZUL for v in dados]

    fig, eixo = plt.subplots(figsize=(8, 5.5))
    barras = eixo.barh(dados.index, dados.abs(), color=cores, height=0.7)
    for barra, valor in zip(barras, dados):
        eixo.text(barra.get_width() + 0.01, barra.get_y() + barra.get_height() / 2,
                  f"{valor:+.2f}", va="center", fontsize=8, color=TEXTO_2)

    eixo.set_xlim(0, 0.8)
    eixo.set_xlabel("força da correlação com o RUL (valor absoluto)")
    eixo.grid(axis="y", visible=False)
    eixo.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=LARANJA), plt.Rectangle((0, 0), 1, 1, color=AZUL)],
        labels=["sobe perto da falha", "desce perto da falha"],
        loc="lower right", frameon=False,
    )
    eixo.set_title(
        "Sensores que mais acompanham o desgaste\ndados simulados NASA C-MAPSS FD001",
        loc="left", fontsize=11, color=TEXTO,
    )
    fig.tight_layout()
    caminho = FIGURAS / "02_correlacao_com_rul.png"
    fig.savefig(caminho, dpi=130)
    plt.close(fig)
    return caminho


def main() -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    df = carregar()

    uteis, constantes = separar_sensores(df)
    print(f"Sensores constantes ({len(constantes)}), descartados:")
    for s in constantes:
        print(f"  - {s}")

    corr = correlacao_com_rul(df, uteis)
    print(f"\nSensores que variam ({len(uteis)}), do mais ao menos ligado ao desgaste:")
    for sensor, valor in corr.items():
        direcao = "sobe" if valor < 0 else "desce"
        print(f"  {valor:+.2f}  {sensor:<32} ({direcao} perto da falha)")

    print("\nPares redundantes (correlação entre si ≥ 0,9):")
    for a, b, valor in pares_redundantes(df, uteis):
        print(f"  {a} × {b}: {valor:+.2f}")

    # Comparação simples: motor saudável (início da vida) x motor perto da falha
    saudavel = df[df["rul"] >= 150][uteis].mean()
    fim = df[df["rul"] <= 10][uteis].mean()
    variacao = ((fim - saudavel) / saudavel * 100).loc[corr.index]
    print("\nVariação média entre motor saudável (RUL ≥ 150) e perto da falha (RUL ≤ 10):")
    for sensor, pct in variacao.items():
        print(f"  {pct:+.3f}%  {sensor}")

    (RAIZ / "dados" / "sensores_uteis.txt").write_text("\n".join(uteis) + "\n", encoding="utf-8")
    f1 = grafico_sensores_ate_falha(df, uteis, corr)
    f2 = grafico_correlacao(corr)
    print(f"\nFiguras salvas em:\n  {f1}\n  {f2}")


if __name__ == "__main__":
    main()
