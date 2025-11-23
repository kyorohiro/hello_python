
```
python3.11 -m venv .venv
source .venv/bin/activate 
pip install --upgrade pip
pip install --upgrade setuptools wheel

pip install sentence-transformers gensim numpy
pip install datasets
pip install 'accelerate>=0.26.0'
pip install annoy
pip freeze > requirements.txt
deactivate
```


```
pip install -r requirements.txt
```
z