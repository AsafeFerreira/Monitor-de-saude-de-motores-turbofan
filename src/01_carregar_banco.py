"""Etapa 1: carrega os arquivos do C-MAPSS (FD001) num banco SQLite.

Cada linha dos arquivos é a leitura de um motor em um ciclo de operação:
id do motor, número do ciclo, 3 configurações de operação e 21 sensores.

No treino, cada motor roda até falhar. Por isso a vida útil restante (RUL)
de cada linha é: último ciclo daquele motor - ciclo atual.
No teste, a série para antes da falha, e o RUL real do último ciclo de cada
motor vem no arquivo RUL_FD001.txt.
"""
import sqlite3
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
BRUTOS = RAIZ / "dados" / "brutos"
BANCO = RAIZ / "dados" / "motores.db"

# Nomes dos sensores conforme a documentação do conjunto de dados
SENSORES = [
    "t2_temp_entrada_fan", "t24_temp_saida_lpc", "t30_temp_saida_hpc", "t50_temp_saida_lpt",
    "p2_pressao_entrada_fan", "p15_pressao_bypass", "p30_pressao_saida_hpc",
    "nf_rotacao_fan", "nc_rotacao_nucleo", "epr_razao_pressao_motor",
    "ps30_pressao_estatica_hpc", "phi_razao_combustivel_ps30",
    "nrf_rotacao_fan_corrigida", "nrc_rotacao_nucleo_corrigida", "bpr_razao_bypass",
    "farb_razao_combustivel_ar_queimador", "htbleed_entalpia_sangria",
    "nf_dmd_rotacao_fan_demandada", "pcnfr_dmd_rotacao_fan_corrigida_demandada",
    "w31_sangria_refrigeracao_hpt", "w32_sangria_refrigeracao_lpt",
]
COLUNAS = ["motor", "ciclo", "config_1", "config_2", "config_3"] + SENSORES


def ler(arquivo: str) -> pd.DataFrame:
    df = pd.read_csv(BRUTOS / arquivo, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLUNAS)]  # os arquivos têm espaços sobrando no fim da linha
    df.columns = COLUNAS
    return df


def main() -> None:
    treino = ler("train_FD001.txt")
    teste = ler("test_FD001.txt")

    # RUL do treino: quantos ciclos faltam até a falha em cada leitura
    ultimo_ciclo = treino.groupby("motor")["ciclo"].transform("max")
    treino["rul"] = ultimo_ciclo - treino["ciclo"]

    rul_teste = pd.read_csv(BRUTOS / "RUL_FD001.txt", header=None, names=["rul_real"])
    rul_teste.insert(0, "motor", range(1, len(rul_teste) + 1))

    with sqlite3.connect(BANCO) as con:
        treino.to_sql("leituras_treino", con, if_exists="replace", index=False)
        teste.to_sql("leituras_teste", con, if_exists="replace", index=False)
        rul_teste.to_sql("rul_teste", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_treino_motor ON leituras_treino(motor, ciclo)")
        con.execute("CREATE INDEX IF NOT EXISTS ix_teste_motor ON leituras_teste(motor, ciclo)")

        resumo = pd.read_sql(
            """
            SELECT COUNT(*)            AS motores,
                   MIN(vida)           AS vida_minima,
                   ROUND(AVG(vida), 1) AS vida_media,
                   MAX(vida)           AS vida_maxima
            FROM (SELECT motor, MAX(ciclo) AS vida FROM leituras_treino GROUP BY motor)
            """,
            con,
        )

    print(f"Banco criado em {BANCO}")
    print(f"Treino: {len(treino)} leituras | Teste: {len(teste)} leituras | RUL de teste: {len(rul_teste)} motores")
    print("Vida dos motores no treino (em ciclos até a falha):")
    print(resumo.to_string(index=False))


if __name__ == "__main__":
    main()
