"""Prepara a pasta web/ para publicar o painel como site estático (stlite).

Dados: NASA C-MAPSS, subconjunto FD001 (simulados e públicos, não são dados da GE).

Como funciona a versão web:
  O stlite é o Streamlit compilado para rodar DENTRO do navegador do visitante
  (via Pyodide, o Python em WebAssembly). Não existe servidor rodando Python:
  o site só entrega arquivos (HTML, o código do painel e o banco), e o navegador
  baixa o Python e executa tudo localmente. Por isso dá para hospedar no GitHub
  Pages ou em qualquer hospedagem de site estático.

O que este script faz:
  1. Copia o painel (src/04_painel.py) para web/src/.
  2. Cria um banco REDUZIDO em web/dados/motores.db, só com o que o painel lê.
     O banco completo tem 7 MB; o visitante baixaria tudo isso sem necessidade.
  3. Copia a configuração visual do Streamlit (.streamlit/config.toml) com o nome
     streamlit-config.toml, porque hospedagens de site costumam bloquear pastas que
     começam com ponto. O index.html monta o arquivo de volta em .streamlit/ no navegador.

Rodar depois das etapas 1 a 3:
    .venv/bin/python src/exportar_web.py
Testar no navegador:
    .venv/bin/python -m http.server 8000 --directory web   →  http://localhost:8000
"""
import shutil
import sqlite3
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
BANCO = RAIZ / "dados" / "motores.db"
WEB = RAIZ / "web"
BANCO_WEB = WEB / "dados" / "motores.db"

# Mesmos sensores mostrados no detalhe do painel (SENSORES_DETALHE em 04_painel.py)
SENSORES = [
    "ps30_pressao_estatica_hpc",
    "t50_temp_saida_lpt",
    "nc_rotacao_nucleo",
    "phi_razao_combustivel_ps30",
]


def main() -> None:
    (WEB / "src").mkdir(parents=True, exist_ok=True)
    (WEB / "dados").mkdir(parents=True, exist_ok=True)

    shutil.copy(RAIZ / "src" / "04_painel.py", WEB / "src" / "04_painel.py")
    shutil.copy(RAIZ / ".streamlit" / "config.toml", WEB / "streamlit-config.toml")

    BANCO_WEB.unlink(missing_ok=True)  # começa do zero a cada exportação
    colunas = ", ".join(SENSORES)
    with sqlite3.connect(BANCO_WEB) as web:
        # ATTACH abre o banco completo dentro da mesma conexão, com o apelido "completo",
        # e assim copiamos tabelas de um banco para o outro com SQL puro.
        web.execute("ATTACH DATABASE ? AS completo", (str(BANCO),))
        web.execute("CREATE TABLE previsoes_teste AS SELECT * FROM completo.previsoes_teste")
        web.execute(f"CREATE TABLE leituras_teste AS SELECT motor, ciclo, {colunas} FROM completo.leituras_teste")
        # Do treino, o painel só usa as médias de motores saudáveis (RUL >= 125) e perto
        # da falha (RUL <= 10). Guardamos só essas linhas e colunas.
        web.execute(
            f"CREATE TABLE leituras_treino AS SELECT rul, {colunas} FROM completo.leituras_treino "
            "WHERE rul >= 125 OR rul <= 10"
        )
        web.execute("CREATE INDEX ix_teste_motor ON leituras_teste(motor, ciclo)")
        web.commit()
        web.execute("DETACH DATABASE completo")
        web.execute("VACUUM")  # compacta o arquivo

    tamanho = BANCO_WEB.stat().st_size / 1024
    print(f"Pasta web/ pronta. Banco reduzido: {tamanho:.0f} KB (o completo tem {BANCO.stat().st_size / 1024:.0f} KB)")
    print("Teste local: .venv/bin/python -m http.server 8000 --directory web  →  http://localhost:8000")


if __name__ == "__main__":
    main()
