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
the most simple, assuming $$n$$ binary inputs and a single binary output which
takes the value $$1$$ when at least $$k$$ of the input values are $$1$$ and $$0$$
otherwise. We can write this as a function  $$AN : \mathbf{Z}_2^n \rightarrow \mathbf{Z} $$where $$ \mathbf{Z}_2 = \{0,1\}$$
satisfying:

$$
\begin{equation}\label{AN1}\tag{AN1}
AN(x) = H_k(\mathbf{1}^T x),
\end{equation}
$$

where $$\mathbf{1}$$ denotes the $$n$$-vector of ones, and $$H_k$$ denotes the heaviside
function at $$k$$ ($$H(z) = 1$$ if $$z \geq k$$ else $$0$$). 

This does not seem so interesting in of itself, however the idea of combining
many simple $$AN$$ units in some hierarchical structure, or _architecture_, is the motivation
behind many deep learning systems today.

## The Perceptron

Introduced by [Rosenblatt, F. \[1957\]](/sections/references/#Rosenblatt_F),
the Perceptron is one of the simplest architectures involving multiple artificial nuerons. The
nueron structure used by Rosenblatt, slightly different from that of
McCullogh and Pitts (\ref{AN1}), is called a linear threshold unit (LTU). The key
difference is that each LTU takes input from $$\mathbf{R}^n$$, and is
associated with an $$n$$-dimensional parameter, often called the _weights_ vector. Thus an
$$LTU_w : \mathbf{R}^n \rightarrow \mathbf{Z}_2$$ can be written as:

$$
\begin{align}\tag{AN2}\label{AN2}
LTU_w(x) = H(w^T x),
\end{align}
$$
 
where $$w$$ is the real valued parameter or weight vector, and $$H$$ is the
heaviside function at $$0$$.

A _Perceptron_ is a set of LTUs arranged in a _layer_, each taking the same input, and giving
a unique output. In this way, a Perceptron is the multivariate response version of an LTU. 
A Perceptron consisting of $$k$$ LTU units can be defined as 
$$P_{w_1,\dots,w_k} : \mathbf{R}^n \rightarrow \mathbf{Z}_2^k$$ with:

$$
\begin{equation}\tag{P1}\label{P1}
P_{w_1, \dots, w_k}(x) = \left(H(w_1^Tx), \dots, H(w_k^T x)\right).
\end{equation}$$

Parameterised this way, it is not so hard to see the analogy between this
structure and that of common multivariate response statistical models. For
example, replacing $$H$$ with the identity function gives the form of
multivariate linear regression, where $$w_i$$ represents the coefficients (typically $$\beta_i$$)
and $$x$$ denotes a feature vector. Similarly, replacing $$H$$ with $$e^{w^T x}$$
yields a similar (up to normalising constant) structure to that of the multinomial logit model. 
The key conceptual difference
is that statistical models are always derived from some underlying assumptions about
the distribution of the data, while Perceptrons, and more generally machine
learning models, are not. 

More succinctly, the weight vectors, $$w_i$$ for $$i \in 1 \leq i \leq k$$, can be inserted as columns
inta a weights matrix $$W \in \mathbf{R}^{n \times k}$$. Equation (\ref{P1})
can then be rewritten as:

$$
\begin{align}\label{P2}\tag{P2}
P_W(x) = \phi( x W), &&\mathrm{where} \,\,\,\, \phi(\cdot) = H(\cdot).
\end{align}
$$

Note that $$\phi$$ is usually referred to as the _activation function_, 
and in general does not have to be a step function, although in Rosenblatt's
implementation it is.

One final notational adjustment, is that typically these models are specified with
an extra parameter called a _bias_, $$b$$, which is more or less equivalent to
an intercept term in statistical modeling. It could be represented by appending
a $$1$$ to each feature vector, and appending a column, $$b$$, to the weight
matrix $$W$$. However, more commonly it is represented as below: 

$$
\begin{align}\tag{P3}\label{P3}
P_{W,b}(x) = \phi( x W + b), &&\mathrm{where} \,\,\,\, \phi(\cdot) = H(\cdot).
\end{align}
$$

One reason it is more convenient to include the bias term $$b$$ explicitly,
instead of appending a $$1$$ to each feature vector, is that in multilayer
networks it is common to add a bias term at each layer, which would not be
straightforward to achieve by editing the feature vector at input layer only.


## Perceptron as a classifier

The Perceptron as defined in (\ref{P3}) above is a linear classifier. It maps a feature vector $$x$$ to a 
$$k$$-vector of binary outputs, by computing a weighted sum of the inputs and applying a step function.

### Single class

The $$2$$-class case ($$k=1$$) is straightfoward; given an input $$x \in \mathbf{R}^n$$, $$P_{W,b}(x)$$ is
either $$0$$ or $$1$$, each value representing one of the two classes.
Geometrically, the Perceptron with $$k=1$$ (just a single LTU) defines
a hyperplane in the feature space, which acts as the _decision boundary_. The hyperplane is defined by:

$$
\begin{align}\tag{H1}\label{H1}
\{x \in \mathbf{R}^n : w^T x + b = 0\}
\end{align}
$$

Every point lying on one side of the hyperplane will be of one
class, and every point on the other side will be the other. To understand which
side of the hyperplane corresponds to each class, consider an arbitrary point
$$p$$ lying on that hyperplane. Since it is on the hyperplane, $$w^T x + b =0$$.
Now move a small amount away from $$p$$ in some direction $$\epsilon \in \mathbf{R}^n$$, 
and note that $$P_{W,b}(x + \epsilon) = w^T (x + \epsilon) + b = w^T \epsilon$$, since $$x$$ was already on
the plane. Therefore the class of this new point is determined by the sign of
$$w^T\epsilon$$, which (asuming $$w$$ is normalised so that $$||w||=1$$, without
loss of generality) is exactly the length of the component of $$\epsilon$$ in the
direction of $$w$$. In other words, the direction of the weight vector $$w$$,
which is by definition perpendicular to the hyperplane it defines, determines
which class either side of the hyperplane will be. 

### Multiple classes 

To generalise this to the multiclass setting ($$k \geq 2$$), some extra work is
required. The $$m^{th}$$ LTU is trained to return $$1$$ if a point is in class
$$m$$, and $$0$$ otherwise - this is known as _1 vs the rest_ approach.
However, this alone is not sufficient, since in general it is possible for
multiple classes be predicted simultaneously for a single point, or no classes
at all.
To address these issues, the _maximum margin_ approach is
used. 

The idea behind the _maximum margin_ approach is that the distance between
a point and the decision boundary can be used as a proxy for confidence. For
example, if both the $$i^{th}$$ and $$j^{th}$$ LTU return $$1$$ for a given
point $$x$$, then to decide which class is more appropriate, the distance
between each of the corresponding decision boundaries should be considered. If
$$x$$ is very close to decision boundary $$i$$, but far away from decision
boundary $$j$$, then one may conclude the model is more confident $$x$$ should
be in class $$j$$. Similarly, if some point is not classified into any classes
(ie. all LTUs return $$0$$), then the same logic suggests that this point
should be classified into the class it was closest to. This logic can be
formalised by defining the multiclass perceptron
$$MP_{W,b}: \mathbf{R}^n \rightarrow \{0, 1, \dots,
k-1\}$$ as 

$$
\begin{align}\tag{M1}\label{M1}
MP_{W,b} = \underset{0 \leq i \leq k-1}{\mathrm{argmax}} \,\,\, \{ w_i^t x + b_i\}
\end{align}
$$

where $$w_i$$ and $$b_i$$ are the fitted weights and bias respectively from the
$$i^{th}$$ LTU. 

### Deriving the margin

To see that the quantity $$w^T p + b$$ gives the orthonormal (signed) distance
between the hyperplane defined by $$w^T x + b = 0$$ and some point $$p$$,
consider the following. 

Let $$H = \{x \in \mathbf{R}^n : w^T x + b = 0\}$$ be
the decision boundary with $$||w||=1$$, let $$x_0$$ be an arbitrary point on
the boundary, and let $$p$$ be an arbitrary point in $$\mathbf{R}^n$$. Let the
signed orthonormal distance between $$p$$ and $$H$$ be denoted by $$d$$. Now
note $$d$$ is equal to the dot product betwwen the vector from $$x_0$$ to $$p$$
and $$w$$, by definition of the dot product. This can be written as:

$$
\begin{align}
d &= w^T (p - x_0) \\
&= w^T p - w^T x_0 \\
&= w^T p + b, &&\textrm{since }x_0 \in H
\end{align}
$$

as claimed.

## Perceptron Learning Algorithm

The Perceptron Learning Algorithm, or PLA, is a stochastic gradient descent
algorithm for fitting the Perceptron defined by (\ref{P3}). It is stated below,
and then some discussion about its motivation and shortcomings is provided.

### Algorithm: PLA

Given initial weights $$W$$ with columns $$w_1, \dots, w_k$$, bias vector $$b$$, 
and observed data $$\{ (x_i, y_i) \in \mathbf{R}^p \times \mathbf{Z}_2^k: 1 \leq i \leq n\}$$, 
iterate the following steps until stopping criterion is met:

 * __FOR__ $$i$$ in $$\{1, \dots, k \}$$; __DO__
  1. $$W_{:,i} \leftarrow $$
  2. Don't

## Example

The code block below generates some points on the plane and assigns labels in
such a way that the classes are linearly separable.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
```


### Perceptron Learning Algorithm (todo)

* specify algorithm
* add proof of convergence (under linear separability assumption)

### Limitations (todo)

 * Does not output class probabilities
 * Only works if data is linearly separable

## Implementation (todo)

 * Python - sklearn
 * Python - custom
 * julia - custom
