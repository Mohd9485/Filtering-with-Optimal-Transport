# Nonlinear Filtering with Brenier Optimal Transport Maps

This repository is by Mohammad Al-Jarrah, Niyizhen Jin, [Bamdad Hosseini](https://bamdadhosseini.org/), [Amirhossein Taghvaei](https://www.aa.washington.edu/facultyfinder/amir-taghvaei) and contains the Python source code to reproduce the experiments in our 2024 paper [Nonlinear Filtering with Brenier Optimal Transport Maps](https://openreview.net/forum?id=blzDxD6bKt). 


We perform several numerical experiments to study the performance of the OT approach in comparison with the EnKF and SIR algorithms.  The OT algorithm consists of solving a max-min problem over function classes $f\in \cal F, T\in \cal T$ with $f, T$ 
taken to be resdiual neural nets. The network weights are learned with a gradient ascent-descent procedure using the Adam optimization algorithm. To reduce the computational cost, the optimization iteration number decreases as the time grows because the OT map is not expected to change significantly from a time step to the next one.


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


