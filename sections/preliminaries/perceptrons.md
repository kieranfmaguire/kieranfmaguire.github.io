---
title: "Preliminaries / Perceptrons"
layout: default
---

# Overview

This section introduces artificial nuerons and perceptrons. It is roughly based on Chapter 10, ([Geron, A. \[2019\]](/sections/references/#Geron_A)),  

## Artifical Neurons

Artificial nuerons (or just nuerons when context is clear) were
originally inspired by their biological counterparts. The first version, proposed by
 [McCulloch, W. & Pitts, W \[1943\]](/sections/references/#McCulloch_Pitts), was intended to directly model 
the biological version. There have been many variations proposed since then,
but in general they are based on the idea that a neuron is a single unit in
a larger network, which
accepts some input, and produces some output. The version proposed by McCulloch & Pitts was
the most simple, assuming $n$ binary inputs and a single binary output which
takes the value $1$ when at least $k$ of the input values are $1$ and $0$
otherwise. We can write this as a function  $ AN : \mathbf{Z}_2^n \rightarrow \mathbf{Z} $ where $ \mathbf{Z}_2 = \\{0,1\\}$
satisfying:

$$
\begin{equation}
AN(x) = H_k(\mathbf{1}^T x),
\end{equation}$$

where $\mathbf{1}$ denotes the $n$-vector of ones, and $H_k$ denotes the heaviside
function at $k$ ($H(z) = 1$ if $z \geq k$ else $0$). 

This does not seem so interesting in of itself, however the idea of combining
many simple $AN$ units in some hierarchical structure, or _architecture_, is the motivation
behind many deep learning systems today.

## The Perceptron

Introduced by [Rosenblatt, F. \[1957\]](/sections/references/#Rosenblatt_F),
the Perceptron is one of the simplest architectures involving multiple artificial nuerons. The
nueron structure used by Rosenblatt, slightly different from that of
McCullogh and Pitts, is called a linear threshold unit (LTU). The key
difference is that each LTU unit takes input from $\mathbf{R}^n$, and is
associated to an $n$-dimensional parameter, often called the _weights_ vector. Thus an
$LTU_w : \mathbf{R}^n \rightarrow \mathbf{Z}_2$ can be written as:

$$ LTU_w(x) = H(w^T x), $$
 
where $w$ is the real valued parameter or weight vector, and $H$ is the
heaviside function at $0$.

A _Perceptron_ is a set of LTUs arranged in a _layer_, each taking the same input, and giving
a unique output. In this way, a Perceptron is the multivariate response version of an LTU. 
A Perceptron consisting of $k$ LTU units can be defined as 
$P_{w_1,\dots,w_k} : \mathbf{R}^n \rightarrow \mathbf{Z}_2^k$ with:

$$P_{w_1, \dots, w_k}(x) = \left(H(w_1^Tx), \dots, H(w_k^T x)\right).$$

Parameterised this way, it is not so hard to see the analogy between this
structure and that of common multivariate response statistical models. For
example, replacing $H$ with the identity function gives the form of
multivariate linear regression, where $w_i$ represents the coefficients (typically $\beta_i$)
and $x$ denotes a feature vector. Similarly, replacing $H$ with $e^{w^T x}$
yields a similar (up to normalising constant) structure to that of the multinomial logit model. 
The key conceptual difference
is that statistical models are always derived from some underlying assumptions about
the distribution of the data, while Perceptrons, and more generally machine
learning models, are not. 

More commonly, Perceptions are represented as 
