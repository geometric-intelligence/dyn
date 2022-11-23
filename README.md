# Regression-Based Elastic Metric Learning #

Official PyTorch implementation of the paper “Regression-Based Elastic Metric Learning on Shape Spaces of Cell Curves”

***[[Paper](https://arxiv.org/abs/2207.12126)] published at [[NeurIPS Workshop Learning Meaningful Representations of Life](https://www.lmrl.org/)]***

![Overview of REML's goal and results](/images/summary_fig.pdf)

Regression-Based Elastic Metric Learning is a machine learning tool designed to improve analysis of 2D curves changing over time. We consider the elastic metric, which is parameterized by $a$ and $b$ which quantify how much two shapes are "stretched" or "bent" compared to each other, respectively. Changing $a$ and $b$ of the elastic metric changes the distance between various points on the manifold of discrete curves: the space where we analyze curves. As such, changing $a$ and $b$ changes the nature of geodesics on the manifold of discrete curves. Our paradigm learns the $a$ and $b$ which have a geodesic closest to the input data trajectory, as evaluated by the coefficient of determination $R^2$. Learning these parameters allows us to then perform geodesic regression on the data trajectory in a space where the data trajectory is closest to a geodesic of that space -- thus improving regression fit and predictive power.

We apply our paradigm to data trajectories of cell outlines changing over time. Cells are dynamic objects that change their shape as they move and evolve. Cell shape is indicative of cell function, so analyzing the shape of a cell over time can provide key insights to the internal evolutions of the cell.

## 🌎 Bibtex ##
If this code is useful to your research, please cite:

```
@misc{https://doi.org/10.48550/arxiv.2210.01932,
  doi = {10.48550/ARXIV.2210.01932},

  url = {https://arxiv.org/abs/2210.01932},

  author = {Myers, Adele and Miolane, Nina},

  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},

  title = {Regression-Based Elastic Metric Learning on Shape Spaces of Elastic Curves},

  publisher = {arXiv},

  year = {2022},

  copyright = {Creative Commons Attribution 4.0 International}
}
```

## 🏡 Installation ##

This codes runs on Python 3.8. We recommend using Anaconda for easy installation. To create the necessary conda environment, run:
```
cd dyn
conda env create -f environment.yml
conda activate dyn
```

## 🏃‍♀️ Run the Code ##

We use Wandb to keep track of our runs. To launch a new run, follow the steps below.

#### 1. Set up [Wandb](https://wandb.ai/home) logging.

Wandb is a powerful tool for logging performance during training, as well as animation artifacts. To use it, simply [create an account](https://wandb.auth0.com/login?state=hKFo2SBNb0U4SjE0ZWN3OGZtbTlJWTRpYkNmU0dUTWZKSDk3Y6FupWxvZ2luo3RpZNkgODhWd254WW1zdG51RTREd0pWOGVKWVVzZkVOZ0dydGqjY2lk2SBWU001N1VDd1Q5d2JHU3hLdEVER1FISUtBQkhwcHpJdw&client=VSM57UCwT9wbGSxKtEDGQHIKABHppzIw&protocol=oauth2&nonce=dEZVS3dvYXFVSjdjZFFGdw%3D%3D&redirect_uri=https%3A%2F%2Fapi.wandb.ai%2Foidc%2Fcallback&response_mode=form_post&response_type=id_token&scope=openid%20profile%20email&signup=true), then run:
```
wandb login
```
to sign into your account.

#### 2. Create a new project in Wandb.

Create a new project in Wandb called "metric-learning-TEST".

#### 3. Specify hyperparameters in default_config.py.

"main.py" of our program runs through every combination of hyperparameters specified in "default_config.py". Change any of the following hyperparameters:
- a_true: "a" is the metric parameter used to generate the synthetic shape trajectory. This is the metric parameter that the code is trying to learn.
- n_sampling_points: the number of points in each cell shape
- n_times: the number of cell shapes in the data trajectory
- noise_std: the amount of noise added to the synthetic data. (how "noisy" is each cell shape)
- percent_train: percent of the data trajectory used to train the regression model in our code
- percent_val: percent of the data trajectory used to validate the regression model (using the coefficient of determination $R^2$) and learn a* (our code's best estimate of a_true). Note: the rest of the data trajectory that is not used to train or validate the model will be used to test the predictive power of a* regression against the baseline square-root-velocity (SRV) metric regression.
- dataset: you can either test the code on a synthetic geodesic between a circle and an ellipse or a semi-synthetic geodesic between two real cancer cells.

#### 4. Run!
For a single run, use the command:
```
python main.py
```
This will initiate runs with every combination of hyperparameters detailed in default_config.py.

### 5. 👀 See Results.

You can see all of your runs by logging into the Wandb webpage and looking under your project name "metric-learning-TEST". Our code automatically names each run as

{dataset-used}_{a_true}_{gradient-ascent-initiation}_mt1_{n_times}_{n_sampling_points}_{noise_std}_{time-of-run}

## 👩‍🔧 Authors ##
[Adele Myers](https://ahma2017.wixsite.com/adelemyers)

[Nina Miolane](https://www.ninamiolane.com/)
