```
conda env remove --name rna-novo
conda install -n base -c conda-forge mamba 

mamba create --name rna-novo python=3.12 gensim numpy scipy pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge
mamba install scipy=1.12
mamba install plotly seaborn pandas matplotlib tqdm scikit-learn kagglehub -c conda-forge

mamba install transformers -c conda-forge
mamba install tensorboard



mamba create -n rna python=3.12 \
scipy=1.12 \
gensim \
numpy \
pandas \
scikit-learn \
matplotlib \
seaborn \
plotly \
tqdm \
kagglehub \
transformers \
tensorboard \
pytorch \
torchvision \
torchaudio \
pytorch-cuda=12.1 \
-c pytorch \
-c nvidia \
-c conda-forge
```

```
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

python -c "import gensim; print(f'Gensim version: {gensim.__version__}')"
```