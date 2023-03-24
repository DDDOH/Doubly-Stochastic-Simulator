# Doubly Stochastic Simulator

Status: Actively maintained. Any question or suggestion is welcome. Contact us by [email](mailto:yufeng_zheng@berkeley.edu?subject=[GitHub]%20Doubly%20Stochastic%20Simulator) or raise an issue. We will give response within 48 hours.

Code and dataset for the paper "A Doubly Stochastic Simulator with Applications in Arrivals Modeling and Simulation".

# Requirements
We recommend using Anaconda to manage the environment. The code is tested on the latest version of python and pytorch (2023, March 24).

The following packages are required:
```bash
conda create --name dsc
conda activate dsc
conda install pytorch -c pytorch -y
conda install -c conda-forge jupyterlab -y
conda install numpy -y
conda install pandas -y
conda install -c conda-forge matplotlib -y
pip install progressbar2
conda install -c anaconda scipy -y
conda install -c conda-forge colored -y
conda install -c anaconda seaborn -y
```

We provide two ways to train the generative model. The first one is to use a discriminator, where code is provided in `train_gan.py`. The second one is to use sinkhorn distance, with code in `train.py`. The second method runs faster but may not be as stable as the first one. If you want to use the second method, you need to install the following package:
```bash
pip install geomloss
```

For the oakland call center dataset, please download it from [Kaggle](https://www.kaggle.com/datasets/cityofoakland/oakland-call-center-public-work-service-requests). After unzip the file, you should have a folder named `service-requests-received-by-the-oakland-call-center.csv`. Put the file in `dataset/callcenter/`.

# Run Experiments

The experiment consists of three steps: prepare dataset, train model, evaluate model. We highlight the steps for reproducing the results on call center dataset. The steps for other datasets are similar.

1. Prepare dataset: run the jupyter notebook in dataset/callcenter_dataset.ipynb.
2. Train model: run train.py with the corresponding exp_label. Specifically,
```bash
python train_gan.py --exp_label callcenter_gan_0
```
3. Evaluate model: 
```bash
python evaluate/bimodal_callcenter_evaluate.py --exp_label callcenter_0
```
For experiments on other dataset, please refer to the corresponding jupyter notebook in `dataset/`, the comments in `train_gan.py` and `train.py`, and the comments in `evaluate/bimodal_callcenter_evaluate.py`, `evaluate/infinite_server_queue.py` and `evaluate/pgnorta_evaluate.py`.

# Credit
The [call center dataset](https://www.kaggle.com/datasets/cityofoakland/oakland-call-center-public-work-service-requests) and [bike sharing dataset](https://www.kaggle.com/c/bike-sharing-demand/data) are both downloaded from Kaggle.