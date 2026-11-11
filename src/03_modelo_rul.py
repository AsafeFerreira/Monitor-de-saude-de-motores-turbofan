"""Etapa 3: modelo que prevê a vida útil restante (RUL) de cada motor.

Dados: NASA C-MAPSS, subconjunto FD001 (simulados e públicos, não são dados da GE).

Passo a passo:
  1. Monta a tabela de entrada (X = sensores) e o alvo (y = RUL limitado a 125).
  2. Separa 20 motores do treino para validação e compara 3 modelos:
     uma referência boba (sempre chuta a média), regressão linear e floresta aleatória.
  3. Treina o melhor modelo com os 100 motores do treino.
  4. Mede o erro no conjunto de teste oficial (último ciclo de cada motor x RUL real).
  5. Salva as previsões no banco (tabela previsoes_teste), o modelo e duas figuras.

Saídas:
  - tabela previsoes_teste em dados/motores.db  → usada pelo painel (etapa 4) e pelo assistente (etapa 5)
  - dados/modelo_rul.joblib                     → o modelo treinado
  - docs/figuras/03_previsto_vs_real.png
  - docs/figuras/04_importancia_sensores.png
"""
import sqlite3
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
BANCO = RAIZ / "dados" / "motores.db"
FIGURAS = RAIZ / "docs" / "figuras"
MODELO = RAIZ / "dados" / "modelo_rul.joblib"

# Na etapa 2 vimos que os sensores ficam estáveis até uns 100–125 ciclos antes da falha.
# Antes disso, não há sinal de desgaste para o modelo aprender: um motor com RUL 300
# e outro com RUL 150 têm leituras parecidas. Por isso dizemos ao modelo que
# "qualquer RUL acima de 125 conta como 125" (motor saudável).
RUL_MAXIMO = 125

SEMENTE = 42  # fixa o sorteio para o resultado ser o mesmo toda vez que rodar

AZUL = "#2a78d6"
LARANJA = "#eb6834"
CINZA = "#b5b3ad"
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
FUNDO = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "axes.edgecolor": CINZA,
    "axes.labelcolor": TEXTO_2, "xtick.color": TEXTO_2, "ytick.color": TEXTO_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 9,
})


def carregar() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    with sqlite3.connect(BANCO) as con:
        treino = pd.read_sql("SELECT * FROM leituras_treino", con)
        teste = pd.read_sql("SELECT * FROM leituras_teste", con)
        rul_teste = pd.read_sql("SELECT * FROM rul_teste", con)

    sensores = (RAIZ / "dados" / "sensores_uteis.txt").read_text(encoding="utf-8").split()
    # Etapa 2: nc e nrc têm correlação de 0,96 entre si. Ficamos só com nc.
    sensores.remove("nrc_rotacao_nucleo_corrigida")
    return treino, teste, rul_teste, sensores


def erro(real, previsto) -> dict:
    """RMSE: penaliza mais os erros grandes. MAE: erro médio, em ciclos, fácil de explicar."""
    return {
        "RMSE": mean_squared_error(real, previsto) ** 0.5,
        "MAE": mean_absolute_error(real, previsto),
    }


def candidatos() -> dict:
    return {
        # Referência: ignora os sensores e sempre chuta a média do RUL.
        # Um modelo de verdade precisa ganhar dele com folga.
        "Referência (média)": DummyRegressor(strategy="mean"),
        # Reta: RUL = a1*sensor1 + a2*sensor2 + ... + b.
        # O StandardScaler põe todos os sensores na mesma escala antes (média 0, desvio 1),
        # porque a rotação está na casa dos 9000 e a razão de bypass na casa dos 8.
        "Regressão linear": make_pipeline(StandardScaler(), LinearRegression()),
        # Floresta aleatória: 200 árvores de decisão, cada uma treinada com um sorteio
        # diferente das linhas; a previsão final é a média das árvores.
        # Consegue aprender curvas (como a "dobra" perto da falha), coisa que a reta não faz.
        # min_samples_leaf=5: cada folha precisa de pelo menos 5 leituras, o que evita
        # decorar o ruído do treino (overfitting).
        "Floresta aleatória": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEMENTE
        ),
    }


def comparar_na_validacao(treino: pd.DataFrame, sensores: list[str]) -> pd.DataFrame:
    """Separa 20% dos MOTORES (não das linhas) para validação.

    Se sorteássemos linhas, o ciclo 50 do motor 7 poderia cair no treino e o ciclo 51
    na validação. As leituras são quase iguais, então o modelo "colaria" e o erro
    ficaria otimista demais. Separando por motor, a validação simula um motor que
    o modelo nunca viu, que é exatamente o caso real.
    """
    divisor = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEMENTE)
    idx_treino, idx_valid = next(divisor.split(treino, groups=treino["motor"]))
    tr, va = treino.iloc[idx_treino], treino.iloc[idx_valid]

    linhas = []
    for nome, modelo in candidatos().items():
        # O errstate silencia um aviso falso ("divide by zero in matmul") que o numpy
        # dá no macOS ao multiplicar matrizes na regressão linear. As previsões saem
        # corretas (o RMSE dá um número normal, sem NaN), é só o aviso que está errado.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            modelo.fit(tr[sensores], tr["rul"].clip(upper=RUL_MAXIMO))
            previsto = modelo.predict(va[sensores])
        linhas.append({"modelo": nome, **erro(va["rul"].clip(upper=RUL_MAXIMO), previsto)})
    print(f"Validação: {tr['motor'].nunique()} motores para treinar, {va['motor'].nunique()} para validar")
    return pd.DataFrame(linhas).sort_values("RMSE")


def grafico_previsto_vs_real(prev: pd.DataFrame, metricas: dict) -> Path:
    """Cada ponto é um motor de teste. Na linha diagonal, a previsão é perfeita.

    Acima da diagonal, o modelo achou que o motor tem mais vida do que tem (otimista,
    o erro mais perigoso na manutenção). Abaixo, ele foi conservador.
    """
    fig, eixo = plt.subplots(figsize=(6.5, 6))
    otimista = prev["erro"] > 0
    eixo.plot([0, 150], [0, 150], color=TEXTO_2, linewidth=1, linestyle="--")
    eixo.scatter(prev.loc[~otimista, "rul_real"], prev.loc[~otimista, "rul_previsto"],
                 s=36, color=AZUL, edgecolor=FUNDO, linewidth=1, label="conservador (previu menos vida)")
    eixo.scatter(prev.loc[otimista, "rul_real"], prev.loc[otimista, "rul_previsto"],
                 s=36, color=LARANJA, edgecolor=FUNDO, linewidth=1, label="otimista (previu mais vida)")
    eixo.set_xlim(0, 150)
    eixo.set_ylim(0, 150)
    eixo.set_xlabel("RUL real (ciclos)")
    eixo.set_ylabel("RUL previsto (ciclos)")
    eixo.legend(loc="upper left", frameon=False)
    eixo.set_title(
        f"Previsto x real nos 100 motores de teste\n"
        f"erro médio (MAE) = {metricas['MAE']:.1f} ciclos · RMSE = {metricas['RMSE']:.1f}\n"
        f"dados simulados NASA C-MAPSS FD001",
        loc="left", fontsize=10, color=TEXTO,
    )
    fig.tight_layout()
    caminho = FIGURAS / "03_previsto_vs_real.png"
    fig.savefig(caminho, dpi=130)
    plt.close(fig)
    return caminho


def grafico_importancia(modelo: RandomForestRegressor, sensores: list[str]) -> Path:
    """Importância = quanto cada sensor ajudou as árvores a reduzir o erro, somando 1 no total."""
    imp = pd.Series(modelo.feature_importances_, index=sensores).sort_values()
    fig, eixo = plt.subplots(figsize=(7.5, 5))
    barras = eixo.barh(imp.index, imp.values, color=AZUL, height=0.7)
    for barra, valor in zip(barras, imp.values):
        eixo.text(barra.get_width() + 0.005, barra.get_y() + barra.get_height() / 2,
                  f"{valor:.0%}", va="center", fontsize=8, color=TEXTO_2)
    eixo.grid(axis="y", visible=False)
    eixo.set_xlabel("importância no modelo")
    eixo.set_title("Sensores que a floresta aleatória mais usou\ndados simulados NASA C-MAPSS FD001",
                   loc="left", fontsize=11, color=TEXTO)
    fig.tight_layout()
    caminho = FIGURAS / "04_importancia_sensores.png"
    fig.savefig(caminho, dpi=130)
    plt.close(fig)
    return caminho


def main() -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    treino, teste, rul_teste, sensores = carregar()
    print(f"Usando {len(sensores)} sensores. RUL limitado a {RUL_MAXIMO} ciclos.\n")

    # 1) Comparar os modelos em motores separados do treino
    comparacao = comparar_na_validacao(treino, sensores)
    print(comparacao.round(2).to_string(index=False))
    melhor = comparacao.iloc[0]["modelo"]
    print(f"\nMelhor na validação: {melhor}\n")

    # 2) Treinar o melhor com os 100 motores do treino
    modelo = candidatos()[melhor]
    modelo.fit(treino[sensores], treino["rul"].clip(upper=RUL_MAXIMO))

    # 3) Avaliar no teste oficial: só o ÚLTIMO ciclo de cada motor tem RUL real conhecido
    ultimos = teste.sort_values("ciclo").groupby("motor").tail(1).sort_values("motor")
    prev = ultimos[["motor", "ciclo"]].rename(columns={"ciclo": "ultimo_ciclo"})
    prev["rul_previsto"] = np.clip(modelo.predict(ultimos[sensores]), 0, None).round(1)
    prev = prev.merge(rul_teste, on="motor")
    prev["erro"] = (prev["rul_previsto"] - prev["rul_real"]).round(1)

    metricas = erro(prev["rul_real"], prev["rul_previsto"])
    print(f"Teste oficial (100 motores): RMSE = {metricas['RMSE']:.2f} | MAE = {metricas['MAE']:.2f} ciclos")
    print(f"Previsões otimistas (previu mais vida do que tinha): {(prev['erro'] > 0).sum()} motores")
    print(f"Erro maior que 30 ciclos: {(prev['erro'].abs() > 30).sum()} motores")

    # 4) Salvar
    with sqlite3.connect(BANCO) as con:
        prev.to_sql("previsoes_teste", con, if_exists="replace", index=False)
    # compress=3 reduz o arquivo de ~50 MB para alguns MB (200 árvores ocupam espaço)
    joblib.dump({"modelo": modelo, "sensores": sensores, "rul_maximo": RUL_MAXIMO}, MODELO, compress=3)

    f3 = grafico_previsto_vs_real(prev, metricas)
    figuras = [f3]
    if isinstance(modelo, RandomForestRegressor):
        figuras.append(grafico_importancia(modelo, sensores))
    print("\nSalvo: tabela previsoes_teste no banco, " + MODELO.name + " e")
    for f in figuras:
        print(f"  {f}")


if __name__ == "__main__":
    main()
