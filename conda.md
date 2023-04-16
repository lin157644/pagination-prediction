```shell
conda create -n prnsm python=3.9

# Pytorch 0.13.1
# To be compatible with laserembeddings
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
python3 -m pip install laserembeddings
python3 -m laserembeddings download-models

# Tensorflow
conda install -c conda-forge cudatoolkit=11.8.0 -y
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/xslin/miniconda3/pkgs/cuda-nvcc-11.7.99-0' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

python3 -m pip install tensorflow_addons

# Alternative version
# python3 -m pip install tensorflow==2.10.* nvidia-cudnn-cu11==8.6.0.163

# Other
conda install -c conda-forge scikit-learn==0.24.2 matplotlib pandas openpyxl tldextract ipython ipywidgets ipykernel fasttext -y
python3 -m pip install sklearn_crfsuite parsel
python3 -m pip install tensorflow_text
# conda install -c conda-forge nltk -y

# Replace the crf
```

In-case you install the wrong python version.
```shell
# Change python version
conda uninstall python
conda install python=3.9
```

Libcudnn_cnn_infer.so.8 library can not found issue in WSL2
https://github.com/microsoft/WSL/issues/5663#issuecomment-1068499676

libdevice not found at ./libdevice.10.bc [Op:__inference__update_step_xla_1387676]
https://stackoverflow.com/q/75037100

Attempting uninstall: nvidia-cudnn-cu11
    Found existing installation: nvidia-cudnn-cu11 8.7.0.84
    Uninstalling nvidia-cudnn-cu11-8.7.0.84:
      Successfully uninstalled nvidia-cudnn-cu11-8.7.0.84
???????

抓後八個path