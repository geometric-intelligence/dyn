# Regression-Based Elastic Metric Learning #

Official implementation of the paper “Regression-Based Elastic Metric Learning on Shape Spaces of Cell Curves”.

***[[Pre-print](https://arxiv.org/abs/2207.12126)], accepted for publication at [[NeurIPS Workshop Learning Meaningful Representations of Life](https://www.lmrl.org/)]***

![NeurIPS Poster](/images/poster.jpg)

## ⭐️ Overview of Goals ##

- Regression-Based Elastic Metric Learning is a machine learning tool designed to improve analysis of discrete parameterizations of 2D cures changing over time.
- Specifically, we optimize geodesic regression analysis by learning the elastic metric parameters that model a given data trajectory close to a geodesic.


![Overview of REML's goal and results](/images/summary.jpg)
Left: A trajectory may follow a geodesic as calculated by one metric but not follow a geodesic as calculated by another metric. Our paradigm learns the elastic metric (parameterized by a) that best models the data trajectory as a geodesic on the manifold of discrete curves. Right: true cell trajectory overlaid with 1) cells predicted by a regression which utilizes our paradigm’s learned metric parameter (a*) 2) cells predicted by a square-root-velocity (SRV) regression. Regression predictions using the SRV metric (red) do not match the data trajectory (blue), but our algorithm’s a* predicts the data trajectory perfectly: our prediction (green) perfectly overlays the data trajectory (blue).

### Elastic Metric ###
- We consider a family of elastic metrics $\href{https://www.researchgate.net/publication/225134644_On_Shape_of_Plane_Elastic_Curves}{given by}$ $g^{a, b}_c(h, k) = a^2\int_{0}^1\langle D_sh, N\rangle\langle D_sk, N\rangle ds + b^2 \int_{0}^1\langle D_sh, T\rangle\langle D_sk, T\rangle ds$
- We use the elastic metric implementation in
$\href{https://geomstats.github.io/}{Geomstats}$.
- The elastic metric is parameterized by $a$ and $b$ which quantify how much two shapes are "stretched" or "bent" compared to each other, respectively.
- Changing $a$ and $b$ of the elastic metric changes the distance between various points on the manifold of discrete curves: the space where we analyze curves. As such, changing $a$ and $b$ changes the nature of geodesics on the manifold of discrete curves.

!["a" and "b" = "stretching" and "bending"](/images/bend_stretch_operations.jpg)

- Note that the ratio $a/b$ is sufficient to describe variationos of $ g^{a, b}$. Thus, we set $b=0.5$, as varying $b$ only changes units of the calculation.
- Our paradigm learns the $a*$ (and therefore the ratio $a*/b$) which models the data trajectory as being closest to a geodesic, as evaluated by the coefficient of determination $R^2$.
- We use a gradient ascent algorithm, along with our derived analytical expression of $R^2$ in terms of $a$, to find the $a*$ which maximizes $R^2$ for a given trajectory.

### Experiments ###

- We apply our paradigm to data trajectories of cell outlines changing over time
- For each experiement, we generate a semi-synthetic data trajectory by drawing a geodesic between two real cancer cells

![semi-synthetic cell trajectory](/images/synthetic_trajectory.jpg)

- We create the trajectory with a predetermined 1) number of cells 2) number of sampling points (how many times each cell outline is sampled) 3) amount of noise 4) "true $a$" (the metric used to draw the geodesic between real cancer cells). Note that because the semi-synthetic geodesic is drawn with the metric parameter $a_{true}$, the metric parameter $a_{true}$ WILL model the trajectory as a geodesic.
- Thus, the gradient ascent learning scheme aims to learn an $a*$ close to $a_{true}$.


### Results ###
- We compare the predictive power of $a*$ regression to the predictive power of regression with the square-root-velocity (SRV) metric, which is a special case of the elastic metric where $a = 1$ and $b = 0.5$.
- Performing geodesic regression with our learned $a*$ metric parameter improves predictive power to geodesic regression, as geodesic regression is more accurate when the data trajectory is close to a geodesic.

![REML increases predictive power of geodesic regression](/images/ideal_conditions.jpg)

![REML converges a* to a_true and r2 to 1](/images/convergence_results.jpg)


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

```
<dataset-used>_<a_true>_<gradient-ascent-initiation>_mt1_<n_times>_<n_sampling_points>_<noise_std>_<time-of-run>
```

## 👩‍🔧 Authors ##
[Adele Myers](https://ahma2017.wixsite.com/adelemyers)

[Nina Miolane](https://www.ninamiolane.com/)