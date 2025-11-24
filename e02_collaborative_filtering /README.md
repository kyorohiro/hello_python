```
#brew install python@3.9
#python3.9 -m venv .venv309
#source .venv309/bin/activate 
#pip install --upgrade pip
# pip install lightfm scipy numpy

conda create -n lightfm-env python=3.11
conda activate lightfm-env
conda install -c conda-forge lightfm
python -c "from lightfm import LightFM; print('ok')"
conda deactivate
```

