# Nonlinear Filtering with Brenier Optimal Transport Maps

This repository is by Mohammad Al-Jarrah, Niyizhen Jin, [Bamdad Hosseini](https://bamdadhosseini.org/), [Amirhossein Taghvaei](https://www.aa.washington.edu/facultyfinder/amir-taghvaei) and contains the Python source code to reproduce the experiments in our 2024 paper [Nonlinear Filtering with Brenier Optimal Transport Maps](https://openreview.net/forum?id=blzDxD6bKt). 


We perform several numerical experiments to study the performance of the optimal transport (OT) approach in comparison with the ensemble Kalman filte (EnKF) and sequential importance resampling (SIR) algorithms.  The OT algorithm consists of solving a max-min problem over function classes $f\in {\cal F}, T\in \cal T$ with $f, T$ 
taken to be resdiual neural nets. The network weights are learned with a gradient ascent-descent procedure using the Adam optimization algorithm. To reduce the computational cost, the optimization iteration number decreases as the time grows because the OT map is not expected to change significantly from a time step to the next one. Next we present some results from the paper.

## 1. A bimodal static example
Consider the task of computing the conditional distribution of a Gaussian hidden random variable $X \sim N(0,I_n)$ given the observation

$$
\begin{aligned} 
    Y=\frac{1}{2}X\odot X + \lambda_w W, \qquad W \sim N(0,I_n)
\end{aligned}
$$

where $\odot$ denotes the element-wise (i.e., Hadamard) product.

<p align="center">
<img src="/images/squared_static_example.png" width="500" height="350"> <img src="/images/squared_static_example_high_SNR.png" width="500" height="350">
</p>


## 2. A bimodal dynamic example
We consider a dynamic version of the previous example according to the following model:

$$
\begin{aligned}
    X_{t} &= (1-\alpha) X_{t-1} + 2\lambda V_t,\quad X_0 \sim \mathcal{N}(0,I_n)\\
    Y_t &= X_t\odot X_t + \lambda W_t,
\end{aligned}
$$

where $\{V_t,W_t\}_{t=1}^\infty$ are i.i.d sequences of standard Gaussian random variables, $\alpha=0.1$ and $\lambda=\sqrt{0.1}$. The choice of $Y_t$ 
will once again lead to a bimodal posterior $\pi_t$ at every time step.

<p align="center">
<img src="/images/xx_states.png" width="250" height="250"> <img src="/images/xx_mmd.png" width="250" height="250"> <img src="/images/dynamic_example_d_vs_mmd.png" width="250" height="250"> <img src="/images/dynamic_example_N_vs_mmd.png" width="250" height="250">
</p>


## 3. The Lorenz 63 model

## 4. The Lorenz 96 model

## 5. Static image in-painting on MNIST

## 6. Dynamic image in-painting on MNIST


Please consider reading the paper for further details on this example. Also, please consider citing our paper if you find this repository useful for your publication.

```
@inproceedings{
al-jarrah2024nonlinear,
title={Nonlinear Filtering with Brenier Optimal Transport Maps},
author={Mohammad Al-Jarrah and Niyizhen Jin and Bamdad Hosseini and Amirhossein Taghvaei},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=blzDxD6bKt}
}
```


