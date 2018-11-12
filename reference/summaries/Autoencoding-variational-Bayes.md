# Auto-encoding Variational Bayes

## Introduction

We want to perform efficient approximate inference and learning with DGMs whose continuous latent variables and parameters have intractable posterior distribution.

In the article, the authors derive a simple differentiable and unbiased estimator for the lower bound (Stochastic Gradient Variational Bayes, SGVB). This estimator can be optimized using gradient ascent.

In the case of an iid dataset, the authors propose the Auto-Encoding Variational Bayes (AEVB) algorithm. This method enables efficient inference and learning.

*Variational Auto-Encoders*: when a neural network is used for the recognition model.


## Method

In the case of a dataset $X = \{x^{(i)}\}_{i=1}^N$ with $N$ iid samples, whose generation process involves a latent variable $z$.

The process consists of two steps:

1. A value $z^{(i)}$ is generated from some prior distribution $p_{\theta^\*}(z)$ (the latent distribution).

2. A value $x^{(i)}$ is generated following a conditional distribution $p_{\theta^\*}(x \mid z)$

The underlying assumption is that the distributions come from a parametric families and that they are differentiable almost everywhere with respect to $\theta$ and $z$.

The main idea is to find an approximation of the intractable posterior $p_{\theta}(z \min x)$.

We want an algorithm that is efficient even in the case of:

* *Intractability*. The algorithm must work in the case where the marginal likelihood $p_{\theta}(x)$ and the posterior $p_{\theta}(z \mid x)$ are intractable (which prevents the use of **EM**). These intractability are quite common and arise especially in neural networks with non-linear hidden layer.

* *Large dataset*. As such, updating parameters should be possible using a small mini-batch or even a single point.


The authors propose a solution that enables:

1. Efficient approximate ML or MAP estimation of the parameters.

2. Efficient approximate posterior inference. Useful for data representation and coding.

3. Efficient approximate inference of the variable $x$. Useful whenever a prior for $x$ is needed, eg de-noising, in-painting and super-resolution.

In order to solve these problems, the authors introduce a recognition model $q_{\phi}(z \mid x)$, a tractable approximate to the true posterior. It is the **probabilistic encoder**. The likelihood $p_{\theta}(x \mid z)$ can be referred to as the **probabilistic decoder**: given a point $z$ in the latent space (code) it produces a distribution over the possible values of $x$.


### The Variational bound

The marginal log-likelihood is given by (through independence):
$$\log p_{\theta}(x) = \sum_{i=1}^N \log p_{\theta}(x^{(i)})$$

And:
$$\log p_{\theta}(x^{(i)}) = KL(q_{\phi}(z \mid x^{(i)}) || p_{\theta}(z \mid x^{(i)})) + L(\theta, \phi ; x^{(i)})$$

Indeed,
$
\begin{align}
  KL(q_{\phi}(z \mid x^{(i)}) || p_{\theta}(z \mid x)) & = \mathbm{E}_{z \sim q_{\phi}(z \mid x^{(i)})} \left[ \log \frac{q_{\phi}(z \mid x^{(i)})}{p_{\theta}(z \mid x^{(i)})} \right] \\
  & = \mathbm{E}_{z \sim q_{\phi}(z \mid x^{(i)})} \left[ \log q_{\phi}(z \mid x^{(i)})\right] - \mathbm{E}_{z \sim q_{\phi}(z \mid x^{(i)})} \left[ \log p_{\theta}(z \mid x^{(i)})\right] + \log p_{\theta}(x^{(i)})
\end{align}
$

We found a lower bound $L = \mathbm{E}_{z \sim q_{\phi}(z \mid x^{(i)})} \left[ - \log q_{\phi}(z \mid x^{(i)}) + \log p_{\theta}(z \mid x^{(i)})\right]$ for the log-evidence $\log p_{\theta}(x^{(i)})$.

The next step is to differentiate and optimize the lower bound. As is, the usual gradient estimator (MCMC) is a bit problematic as it exhibits very high variance...


### The SGVB estimator and AEVB algorithm
