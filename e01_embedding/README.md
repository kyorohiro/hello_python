python3.11 -m venv .venv
source .venv/bin/activate 
pip install --upgrade pip
pip install --upgrade setuptools wheel

pip install sentence-transformers gensim numpy

pip freeze > requirements.txt
deactivate