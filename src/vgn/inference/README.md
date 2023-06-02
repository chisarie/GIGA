# Installation

```bash
conda create --name giga_env python=3.8
conda activate giga_env
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install scikit-learn networkx==2.5
pip install -r requirements.txt
pip install -e .

python scripts/convonet_setup.py build_ext --inplace
```
