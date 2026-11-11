#!/bin/bash
# Abre o painel no navegador com um único comando:  ./rodar.sh
# Na primeira vez, cria o ambiente Python e instala as bibliotecas.
# Dados: NASA C-MAPSS FD001, simulados e públicos (não são dados da GE).

set -e                      # para o script no primeiro erro
cd "$(dirname "$0")"        # entra na pasta do projeto, de onde quer que seja chamado

if [ ! -d .venv ]; then
    echo "Criando o ambiente Python (.venv)..."
    python3 -m venv .venv
fi

# Instala o que estiver faltando no requirements.txt (é rápido quando já está tudo instalado)
.venv/bin/pip install --quiet --disable-pip-version-check -r requirements.txt

echo "Abrindo o painel em http://localhost:8501  (Ctrl+C para parar)"
# Abre o navegador 4 segundos depois, já com o servidor no ar (o "&" roda isso em paralelo).
# "open" é do macOS; "xdg-open" é o equivalente no Linux.
(sleep 4; open http://localhost:8501 2>/dev/null || xdg-open http://localhost:8501) &

# --server.headless true: pula a pergunta de e-mail que o Streamlit faz na primeira vez
.venv/bin/streamlit run src/04_painel.py --server.headless true
