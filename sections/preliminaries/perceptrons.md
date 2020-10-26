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
P_W(x) = \phi( x^T W), &&\mathrm{where} \,\,\,\, \phi(\cdot) = H(\cdot).
\end{align}
$$

Note that $$\phi$$ is usually referred to as the _activation function_, 
and in general does not have to be a step function, although in Rosenblatt's
implementation it is.

One final notational adjustment, is that typically these models are specified with
an extra parameter called a _bias_, $$b$$, which is more or less equivalent to
an intercept term in statistical modeling. It could be represented by appending
a $$1$$ to each feature vector, and appending an element $$b_i$$ to each weight
vector $$w_i$$.
However, more commonly it is represented as below: 

$$
\begin{align}\tag{P3}\label{P3}
P_{W,b}(x) = \phi( x^T W + b), &&\mathrm{where} \,\,\,\, b\in \mathbf{R}^k.
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
$$i^{th}$$ LTU (where $$i$$ starts at $$0$$ to match class labels). 

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
algorithm for fitting Perceptrons. It is stated below,
and then some discussion about its motivation and shortcomings is provided.

### Algorithm
For notational simplicity, assume the parameterisation described by Equation (\ref{P2}). 
Note this does not lose any generality, as described above. 
Given initial weights $$W$$ with columns $$w_1, \dots, w_k$$,
observed data $$\{ (x_i, y_i) \in \mathbf{R}^p \times \mathbf{Z}_2^k: 1 \leq i \leq n\}$$
and learning rate $$\eta \in \mathbf{R}$$, do the following until stopping
criterion is met:

  * __FOR__ $$i$$ in $$\{1, \dots, n \}$$; __DO__
    * __FOR__ $$j$$ in $$\{1, \dots, k \}$$; __DO__
      * Update weight vector $$j$$ with: $$w_j^{(new)} \leftarrow w_j^{(old)} + \eta(y_{i,j}- \hat{y}_{i,j}) x_i$$

where $$\hat{y}_{i,j}$$ is the $$j^{th}$$ entry of $$P_{W,b}(x_i)$$. Appropriate stopping criterion is typically when $$\hat{y}_{i,j} = y_{i,j}$$ for all $$i,j$$, or
max iterations reached.

### Motivation

One way to understand the algorithm, is to consider the classification of
a single point $$y_{i,j}$$. If $$w_j^T x_i \geq 0$$, then $$\hat{y}_{i,j}
= 1$$, and $$0$$ otherwise. If the corect value of $$y_{i,j}$$ matches
$$\hat{y}_{i,j}$$, then
in the update step for training instance $$i$$, $$w_j$$ will not be
changed since $$(y_{i,j} - \hat{y}_{i,j}) = 0$$. On the other hand, if the
predicted value is incorrect, there are two cases: 

__Case 1:__ $$y_{i,j} = 1 \,\,\mathrm{and}\,\, \hat{y}_{i,j}=0$$  
$$\begin{align}
&\implies y_{i,j} - \hat{y}_{i,j} = 1 \\
&\implies w_j^{(new)} = w_j^{(old)} + \eta x_i \\
&\implies w_j^{(new)\,T}x_i = (w_j^{(old)}+\eta x_i)^Tx_i = w_j^{(old)\,T}x_i + \eta ||x_i||^2 \geq w_j^{(old)\,T}x_i
\end{align}$$ 


__Case 2:__  $$y_{i,j}=0 \,\,\mathrm{and}\,\, \hat{y}_{i,j}=1$$\\
$$\begin{align}
&\implies y_{i,j} - \hat{y}_{i,j} = -1 \\
&\implies w_j^{(new)} = w_j^{(old)} - \eta x_i \\
&\implies w_j^{(new)\,T}x_i = (w_j^{(old)}-\eta x_i)^Tx_i = w_j^{(old)\,T}x_i - \eta ||x_i||^2 \leq w_j^{(old)\,T}x_i
\end{align}$$

In both case $$w_j$$ is changed in such a way that $$w_j^{(new)\,T}x_i$$ is closer to
correctly classifying $$y_{i,j}$$ than previously; if $$y_{i,j}=1$$,
 then $$w_j^{(new)\,T}x_i \geq w_j^{(old)\,T}x_i$$, and vice-versa if
$$y_{i,j} = 0$$.

The same algorithm can also be derived in the framework of stochastic gradient
descent with small notational adjsutment. First, change the class label of $$0$$ to $$-1$$, 
and replace the heaviside step function from Equation (\ref{P2}) with the sign function $$sgn(\cdot)$$.
Also let $$\odot$$ represent elementwise multiplication of two vectors, and let
$$elmax\{a,\, b\}$$ denote the elementwise maximum of vectors $$a$$ and $$b$$.
Then the perceptron loss can be defined as:

$$\begin{align}\tag{L1}\label{L1}
Q_W(x_i) = elmax\{0,\,\, y_i \odot x_i^T W\}
\end{align}$$

An obvious note is that (\ref{L1}) defines a vector of loss values, one for
each class. Typically this is not a useful loss function, since it is ambiguous
what it means to minimise a vector (eg. $$L_0$$ norm and $$L_{\infty}$$ norm
are both valid definitions). However, notice that this formulation actually
reduces to $$k$$ independent minimisation problems (where $$k$$ is the number
of LTUs in the Perceptron); one for each class. In other words, each $$w_j$$
can be fit independently, by minimising the associated scalar loss function:

$$\begin{align}\label{L2}\tag{L2}
Q_{w_j}(x_i) = max\{0,\,\,y_{i,j} w_j^T x_i\}.
\end{align}$$

Differentiating this function with respect to $$w_j$$ gives:

$$\begin{align}\label{L3}\tag{L3}
\nabla \{\,Q_{w_j}(x_i)\, \} &=
  \begin{cases} 
    &0 &&\mathrm{when} \,\, sgn(y_i) = sgn(w_j^Tx_i) \\
    &y_{i,j} \, \nabla \{w_j^T x_i\} &&\mathrm{otherwise} 
  \end{cases} \\
&=
  \begin{cases} 
    &0 &&\mathrm{when} \,\, sgn(y_i) = sgn(w_j^Tx_i) \\
    &y_{i,j} x_i &&\mathrm{otherwise} 
  \end{cases} \\
\end{align}$$

which when plugged into the usual SGD algorithm with step size equal to
$$\eta$$ gives the below formula for updating $$w_j$$:

$$\begin{align}
w_j^{(new)} &= 
\begin{cases}
&w_j^{(old)} &&\mathrm{when} \,\, y_{i,j} = \hat{y}_{i,j} \\
&w_j^{(old)} + \eta y_{i,j} x_i &&\mathrm{otherwise}
\end{cases} \\
&= w_j^{(old)} + \mathbf{1}(y_{i,j}=\hat{y}_{i,j}) \, \eta \,y_{i,j} x_i
\end{align}$$ 

which is cleary equivalent to the formulation given in above.

### Properties

The PLA is only guaranteed to converge when the classes are linearly seperable,
a very strong assumption. Furthermore, when the data is not linearly separable,
the PLA will not even converge towards a reasonable solution as the number of
iterations increases. This limits the usefulness of Perceptrons as a standalone
classifier severely; if it is not known that the classes are linearly separable a priori, Perceptrons should not be used. 
For this reason, it is almost always a better choice to use more modern
algorithms, such as support vector machines (SVM) for example. Nevertheless,
the proof of convergence in the linearly separable case is interesting to
study. It is given below.

### PLA Convergence

As noted above, multiclass problems are reduceable into multiple binary
classification problems, thus the treatment below is only given for the
binary problem. However it is worth noting what it means for the multiclass
problem to be linearly separable. Multiple classes being linearly separable is
actually stronger than just assuming lineary decision boundaries. It means that
 __each of the one vs the rest problems__ must be linearly separable.

Let $$w^{(k)}$$ denote the weights after the $$k^{th}$$ update, and let the
sequence $$t_1, t_2, \dots$$ denote the indices of the observations where each
update occurs. Finally, let the observed data be $$\{(x_i, y_i) \in \mathbf{R}^p \times \{-1,1\} : 1 \leq i \leq n\}$$.
 The proof needs the following three assumptions:

__(A0)__ Initial condition: $$\|w^{(0)}\| = 0$$, where $$\|\cdot\|$$ denotes the $$L_2$$ norm.
 
__(A1)__ Finite data: $$\|x_i\| \leq B < \infty, \,\, \forall i$$

__(A2)__ Linearly separable: $$\exists\,\, w^{\star} \in \mathbf{R}^p : y_i w^{\star \,T}x_i \geq \epsilon, \,\, \forall i$$ for some $$\epsilon > 0$$

Let $$\theta^{(k)}$$ denote the angle between $$w^{\star}$$ and $$w^{(k)}$$, so
cosine of that angle can be written as:

$$\begin{align}\tag{C1}\label{C1}
cos(\theta^{(k)}) = \frac{ w^{\star\,T}w^{(k)} }{ \|w^{\star}\|\|w^{(k)}\| }
\end{align}$$

Studying the numerator of (\ref{C1}), a lower bound can be derived as follows:

$$\begin{align}
w^{(k)} &= w^{(k-1)} + \eta y_{t_k} x_{t_k} \\
\implies w^{\star\,T}w^{(k)} &= w^{\star\,T}w^{(k-1)} + \eta y_{t_k}w^{\star\,T}x_{t_k} \\
\implies  w^{\star\,T}w^{(k)} &\geq  w^{\star\,T}w^{(k-1)} + \eta \epsilon &&\mathrm{by} \,\, \mathrm{(A2)}\\
\implies  w^{\star\,T}w^{(k)} &\geq  w^{\star\,T}w^{(0)} + k \eta \epsilon &&\mathrm{by}\,\,\mathrm{recursion}\\
&= k \eta \epsilon &&\mathrm{by} \,\, \mathrm{(A0)}
\end{align}$$

And for the denomionator, an upper bound can be found by simple application of
triangle inequality and recursively applying the same argument:

$$\begin{align}
\|w^{(k)}\|^2 &= \|w^{(k-1)} + \eta y_{t_k} x_{t_k}\|^2 \\
&\leq \|w^{(k-1)}\|^2 + \eta^2 \|x_{t_k}\|^2 \\
&\leq \|w^{(k-1)}\|^2 + \eta^2 B^2 &\mathrm{by}\,\,\mathrm{(A1)}\\
&\leq k \eta^2 B^2 \\
\implies \|w_{(k)}\| &\leq \sqrt{k} \eta B  
\end{align}$$

Putting the two bounds back into (\ref{C1}) and noting that the cosine function
is bound above by 1 gives the finite bound on $$k$$:

$$\begin{align}
1 &\geq \frac{w^{\star\,T}w^{(k)}}{\|w^{\star}\|\|w^{(k)}\|} \geq \frac{k \eta \epsilon}{\|w^{\star}\| \sqrt{k} \eta B} \\
\implies k &\leq \frac{\|w^{\star}\|^2 B^2}{\epsilon^2},
\end{align}$$

thus proving the result. The bound shows that under the assumptions (A0) - (A2), 
the algorithm will converge in worst case time proportional to the
largest $$\|x_i\|$$, the size of $$\|w^{\star}\|$$ and inversely proportional
to the minimum margin.
Note also that $$\|w^{\star}\|$$ is actually only determined by the entry
corresponding to the bias $$b$$, since w.l.o.g the other entries of $$w$$ can
be normalised without changing the solution (ie. the $$b$$ term is the shift
factor from $$0$$, and the remaining entries of $$w$$ give the direction of the
normal vector defining the orientation of the hyperplane).

## Example

The following code snippets demonstrate the the usage of the Perceptron implementation available
in the Python package [scikit-learn](https://scikit-learn.org/stable/).

### Binary Classification

First, import packagaes that will be needed:
```python
import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
plt.style.use('dark_background')
```

Next, generate some points bin $$\mathbf{R}^2$$, and define a decision boundary
to assign classes. The parameters of the decision boundary are $$w=(1,2)$$ and $$b=-3$$:
```python
# set seed for reproducibility
np.random.seed(2020)

# make some dummy data
x = np.random.randn(100,2)*5
w_star = np.array([1,2])
b_star = 3

# making labels which are linearly seperable
y_star = ((x @ w_star) + b_star > 0).astype(np.int)

# decision boudary
decision_x = np.arange(start=x.min(), stop=x.max()+0.1, step=0.1)
decision_y = -(b_star/w_star[1]) - (w_star[0]/w_star[1])*decision_x

# visualise
fig, ax = plt.subplots()
ax.scatter(x[:,0], x[:,1], c=y_star)
ax.plot(decision_x,decision_y,c='red')
plt.title("Linearly separable classes")
```

The data looks like below; the colour of the points are the
classes, and the red line is the decision boundary defined above.

![](/assets/images/perceptron1.png){: .center-image }

Now, use the Perceptron Learning Algorithm to fit a linear decision boundary to
the data. The fitted decision boundary is slightly different from
the one originally specified, since the PLA does not converge to a unique
solution, just some solution.

```python
# fit a perceptron
p = Perceptron(random_state=2020)
p.fit(x, y_star)

# get decision boundary and accuracy
decision_y_fit = -(p.intercept_[0]/p.coef_[0][1]) - (p.coef_[0][0]/p.coef_[0][1])*decision_x
accuracy = p.score(x, y_star)

# visualise
fig, ax = plt.subplots()
ax.plot(decision_x,decision_y_fit,c='blue',label='fit')
ax.plot(decision_x,decision_y,c='red',label='original')
ax.scatter(x[:,0], x[:,1], c=y_star)
ax.legend()
plt.title(f'Perceptron fit (accuracy={accuracy*100}%)')
```
![](/assets/images/perceptron2.png){: .center-image }

This plot shows quite clearly one of the shortcomings of this method;
intuitively the fitted decision boundary is not ideal. It would make more sense
if the decision boundary were to be equidistant to data on either side. However
to achieve this requires are different loss function, for example maximum
margin or similar.

In the next block, the labels are changed slightly so that the data is no
longer linearly separable.

```python
y_star_2 = y_star.copy()
y_star_2[:5] = 1 - y_star_2[:5]
fig, ax = plt.subplots()
ax.scatter(x[:,0], x[:,1], c=y_star_2)
plt.title("Data no longer linearly separable")
```

![](/assets/images/perceptron3.png){: .center-image }

When the PLA is used to find a decision boundary, it will never converge. To
illustrate, below
chunk computes the decision boundary after every $$100$$ iterations (where an
iteration means one complete pass over the data set, often called an epoch), and
plots them.

```python
# ther is no guarantee the algorithm will converge towards a "good" solution when the data
# is not linearly separable. This function keeps track of error at each iteration, and plots
# the decision boundary every 100 iterations

def fit_and_return_error(num_iters):
    i = 0
    p = Perceptron(random_state=2020,tol=None)
    p.partial_fit(x,y_star_2,classes=np.unique(y_star_2))
    fig, ax = plt.subplots()
    ax.scatter(x[:,0], x[:,1], c=y_star_2)
    plt.title('Fit at different iterations')
    err = []
    while True:
        i += 1
        p.partial_fit(x,y_star_2)
        accuracy = p.score(x,y_star_2)
        err.append(1 - accuracy)
        if i % 100 == 0: 
            dec_y = -(p.intercept_[0]/p.coef_[0][1]) - (p.coef_[0][0]/p.coef_[0][1])*decision_x
            ax.plot(decision_x, dec_y, label=f'iter: {i}')
        if i == num_iters: 
            ax.legend()
            return err
        
num_iters = 500
err = fit_and_return_error(num_iters)
```

![](/assets/images/perceptron4.png){: .center-image }

It doesn't look like the fit at $$500$$ iterations than the fit at $$100$$, as
expected. To confirm, look at the accuracy at each iteration:

```python
plt.plot(np.arange(num_iters), err, label='error')
plt.legend()
plt.title('Error vs number of iterations')
```

![](/assets/images/perceptron5.png){: .center-image }

### Mutliclass classification

Now consider the case when there are 3 linearly separable classes. The block
below generates such data, and produces labels from $$0 - 2$$.

```python
plt.plot(np.arange(num_iters), err, label='error')
plt.legend()
plt.title('Error vs number of iterations')
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
