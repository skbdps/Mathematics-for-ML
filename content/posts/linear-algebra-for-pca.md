---
title: "Essential Linear Algebra for PCA"
date: 2025-02-13
description: "From change of basis to Singular Value Decomposition — the minimal theory you actually need to understand Principal Component Analysis."
tags: ["linear-algebra", "machine-learning", "mathematics"]
series: ["Math for ML"]
draft: false
---

# Essential Linear Algebra for PCA

*This blog assumes that you have a basic knowledge of the coordinate system and their matrix representation and transformation matrix.*
*From change of basis to Singular Value Decomposition — the minimal theory you actually need to understand Principal Component Analysis.*

---

## 1  Why This Post Exists

If you have ever tried to understand Principal Component Analysis (PCA), you have probably run into a wall of notation: eigenvectors, orthogonal matrices, decompositions. Most explanations either drown you in proofs or hand-wave past the intuition. This post takes a different approach. We will build up **only** the linear algebra you truly need, i.e., change of basis, eigenvalues, eigendecomposition, and SVD. Then we will see how each concept feeds into the next until PCA falls out naturally at the end.

## 2  Change of Basis

A **basis** is a set of vectors that can span the entire vector space. In a 2D plane, we generally use the **standard basis**:

$$
e_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \quad e_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

These are the axes you are used to; the horizontal and vertical directions. Every point in the plane can be written as some combination of $e_1$ and $e_2$.

Now, what if I want to use a *different* pair of vectors as my basis? Say:

$$
b_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix} \quad b_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}
$$

Why would we want to do that? The short answer: it can make our calculations *much* easier in many cases. A matrix that looks messy in the standard basis might become beautifully simple in a different one. We will see exactly how later, but first, let us get comfortable with the mechanics.

### An Intuitive Example

Think about what $b_1$ and $b_2$ mean. They are the $[1, 0]$ and $[0, 1]$ of the new basis. Just like $e_1$ and $e_2$ are the "unit rulers" in the standard world, $b_1$ and $b_2$ are the "unit rulers" in the new world. I have used the work unit ruler for understanding, not necessarily unit vectors in standard Euclidean metric.

Now take a random point $[6, 2]^T$ in our original (standard) basis. That point is just $6 e_1 + 2 e_2$. But we can also express it using $b_1$ and $b_2$:

$$
6 e_1 + 2 e_2 = 4 b_1 + 2 b_2
$$

So the same point that was $[6, 2]^T$ in standard coordinates is $[4, 2]^T$ in the new basis. The point itself has not moved at all — we have just changed the ruler we are using to measure it.

> Think of it like currency exchange. ₹500 and \$6 might buy you the same coffee. The coffee did not change, only the numbers on the label did. Change of basis is the same idea for vectors.

### The Change-of-Basis Matrix

We can package this conversion neatly. Stack the new basis vectors as columns to form the **change-of-basis matrix**:

$$
C = \begin{bmatrix} b_1 & b_2 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
$$

Now we have figured out a way to write coordinates of the new basis in our standard basis. If a point has coordinates $\mathbf{v}_{\text{new}}$ in the new basis, its standard coordinates are:

$$
\mathbf{v}_{\text{std}} = C \, \mathbf{v}_{\text{new}}
$$

And $C$ is called the **change-of-basis matrix**. If we want to go the other way — find a point's position in the new basis given its standard coordinates — we just invert:

$$
\mathbf{v}_{\text{new}} = C^{-1} \, \mathbf{v}_{\text{std}}
$$

You can verify: $C^{-1} [6, 2]^T = [4, 2]^T$, exactly what we worked out by hand.

One quick note - If you are feel like the change of basis matrix is just like any other transformation. You would be correct from a purely numerical POV. But, as I have explained what is actually
does, you have to think about the difference in their application. One transforms a single vector, another changes how we describe that measure/represent in our space.

### Transformations in a New Basis

Now let us say we want to calculate what happens when we apply a transformation $T$ from our standard basis to a vector in a new basis.

Here is how we think about it. We have $\mathbf{v}_n$ in the new basis. First, $C \mathbf{v}_n$ converts it to the standard basis. Then, $T C \mathbf{v}_n$ is the transformation applied in the standard basis. Finally, $C^{-1} T C \mathbf{v}_n$ converts the result back to the new basis.

So the transformation in the new basis is:

$$
T_{\text{new}} = C^{-1} T C
$$

The transformation itself has not changed. We have just described it from a different viewpoint. And this raises a natural question: is there a basis where $T$ looks especially simple? Because that is what we were set to do, right? To make our calculation easier.

## 3  Eigenvalues, Eigenvectors, and Eigenbasis

For a square matrix $T$, let us say we assume there exists a vector $\mathbf{v}$ such that:

$$
T\mathbf{v} = \lambda \mathbf{v}
$$

Here $\lambda$ is called the **eigenvalue** and $\mathbf{v}$ is called the **eigenvector** of the transformation $T$.

Now, think about what this equation is really saying. When $T$ acts on most vectors, it does complicated things like rotate them, shear them, scale them, send them off in some completely new direction. But an eigenvector is special: $T$ does *nothing* to it except scale it. The output $T\mathbf{v}$ points in the exact same direction as $\mathbf{v}$, just stretched or compressed by a factor of $\lambda$.

> **Key insight:** The special thing about eigenvectors is *not* that "multiplying a vector by a scalar scales it", that is obviously true for any vector. The special thing is that the *transformation $T$ itself* behaves like scalar multiplication on these particular vectors. Out of all possible directions in the space, eigenvectors are the ones where $T$'s entire complicated action collapses to a single number.

### Intuition: What Eigenvalues Tell Us

Now, what happens when $T$ hits an eigenvector? It will either stretch it or squeeze it (or reflect in case $\lambda$ is negative) but it will not shear it or send it off in some completely random new direction. So the eigenvalue $\lambda$ is simply the amount of scaling that $T$ does along that eigenvector direction.

If you imagine $T$ as physically distorting space (stretching a rubber sheet) then each eigenvector is a direction along which the rubber only gets longer or shorter, never twisted. A $\lambda$ of 3 means that direction triples in length. A $\lambda$ of 0.5 means it halves. If $\lambda = 1$, that direction is untouched. If $\lambda = 0$, $T$ crushes that direction flat — it loses all information there.

### The Eigenbasis

So let us say we have found enough eigenvectors that can span a new basis. Now, what will the transformation $T$ do in that basis?

Well, we just said that $T$ only stretches or squeezes eigenvectors. So if *every* basis vector is an eigenvector, then $T$ can only scale along each basis direction independently. No direction gets mixed with any other.

What kind of matrix only scales along each axis independently? A **diagonal matrix**. And what goes on the diagonal? The amount of scaling in each direction — which is exactly what eigenvalues are.

So in the eigenbasis, $T$ becomes:

$$
T_{\text{eigen}} = \begin{bmatrix} \lambda_1 & 0 & \cdots \\ 0 & \lambda_2 & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}
$$

This is extremely powerful. A matrix that might have been a dense mess of numbers in the standard basis is now completely transparent. Each eigenvalue tells you exactly what $T$ does along one direction, with zero coupling between directions.

### A Useful Example: Computing $T^{100}$

Let us see one helpful use case that also ties together what we have learnt so far.

Say I need to compute $T^{100}$, multiplying $T$ with itself 100 times. In the standard basis, this is a very expensive calculation. But how would we write this transformation in the eigenbasis?

From Section 2, we know $T_{\text{new}} = C^{-1} T C$. If our new basis is the eigenbasis, then $C$ is the matrix of eigenvectors, and $T_{\text{new}}$ is diagonal (as we discussed in the last section). Raising a diagonal matrix to a power is trivial - you just raise each diagonal entry:

$$
T_{\text{eigen}}^{100} = \begin{bmatrix} \lambda_1^{100} & 0 & \cdots \\ 0 & \lambda_2^{100} & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}
$$

Now we can easily switch between our standard and eigenbasis: do the computation in the easy (diagonal) world, then use $C$ to convert the result back. What was an impractical calculation becomes almost free.

## 4  Matrix Decomposition

Now we have built a basic theoretical understanding of change of basis and eigenstuff. Before we move on to PCA, we need one more idea: **matrix decomposition**.

A matrix decomposition is kind of like factorisation for numbers. Just as we can write $12 = 3 \times 4$, we can break a daunting matrix into a product of simpler, easier-to-work-with matrices. We will look at two kinds of decomposition: eigendecomposition and SVD.

### Eigendecomposition

Let us say that an eigenbasis exists for a matrix $A$. Then we can decompose $A$ as:

$$
A = P \Lambda P^{-1}
$$

where $P$ is the matrix whose columns are the eigenvectors, and $\Lambda$ is the diagonal matrix of eigenvalues.

How does this work? Think about it intuitively. When we apply this decomposed matrix onto a vector, first we change our basis to the eigenbasis (that is what multiplying by $P^{-1}$ does). Now the matrix $A$ in the eigenbasis will only do a scaling operation — that is $\Lambda$, the diagonal matrix of eigenvalues. Then we come back to the original basis by multiplying with $P$.

So eigendecomposition says: the action of $A$ on any vector is equivalent to *going into* the eigenbasis, *scaling* there by eigenvalues, and *coming back*. It is a factorisation of $A$ itself — we have broken it into three simple pieces that together reproduce $A$'s full behavior in the standard basis.

**Important limitation:** Eigendecomposition only works for square matrices that are diagonalisable — meaning they have enough independent eigenvectors to form a basis. Not every square matrix qualifies, and rectangular matrices (like a 10,000 × 100 data matrix) are excluded entirely. This is precisely why we need a more general tool. Here is how:

1. Rectangular Matrices (Dimensional Mismatch): An eigenvector equation $Av = \lambda v$ requires the input $v$ and output $Av$ to be parallel. But a rectangular matrix (say, $100 \times 3$) takes a vector from 3D space and shoots it into 100D space. The input and output live in different universes with different dimensions, so they can never be parallel.

2. Defective Matrices (Geometric Collapse): Even some square matrices fail. Think of a "shear" transformation (like pushing the top of a card deck sideways). A shear might leave only one line unchanged, collapsing all other stable directions onto it. If you have a 2D space but only 1 eigenvector, you do not have enough "structural beams" to build a new basis ($P$ becomes non-invertible).This is why we need a more robust tool; one that handles different dimensions and collapsed geometries gracefully. That tool is SVD.

### Singular Value Decomposition (SVD)

Here is a remarkable property of *any* transformation matrix $A$: there exists an orthogonal set of vectors which, when transformed by $A$, remain orthogonal. The directions will change, the lengths will change, but orthogonality is preserved for this special set.

We can write this as:

$$
A v_1 = \sigma_1 u_1 \qquad A v_2 = \sigma_2 u_2 \qquad \ldots
$$

where $v_1, v_2, \ldots$ are orthogonal unit vectors in the input space, $u_1, u_2, \ldots$ are orthogonal unit vectors in the output space, and $\sigma_i = \|Av_i\|$ is how much $A$ stretches the $i$-th direction.

In matrix form, this becomes:

$$
AV = U\Sigma
$$

Now, $V$ is orthogonal, so $V^T = V^{-1}$. Rearranging:

$$
A = U \Sigma V^T
$$

This is the **Singular Value Decomposition**. It works for *any* matrix — square, rectangular, anything.

### The Geometry: Rotation + Scale + Rotation

What does $A = U \Sigma V^T$ mean geometrically? Any transformation is a rotation ($V^T$), followed by a scale ($\Sigma$), followed by another rotation ($U$).

**Why not just rotation + scale?** That should also work, right? Think about it for a moment.

The answer is: the input space and the output space can have different dimensions and different natural orientations. $V^T$ rotates the input to align with the scaling axes. $\Sigma$ does the scaling. Then $U$ rotates the result into the output space's orientation. For a symmetric matrix, the input and output happen to share the same orientation, so $U = V$ and SVD collapses to eigendecomposition. But in general, you need two separate rotations.

There is one decomposition that trivially uses a single rotation: a **scaled identity** ($\alpha I$). It scales every direction equally, so there is nothing to rotate. But that is exactly the problem. It tells us nothing about which directions matter more than others. It is like saying "everything is equally important," which is useless. The whole point of SVD is to reveal that some directions carry far more information, and that requires the full $U \Sigma V^T$ with its two distinct rotations.

### Computing U and V

From $A = U \Sigma V^T$ we can derive:

$$
A^T A = V \Sigma^2 V^T
$$
(We skipped a very small calculation part, try that on your own.)
This is just the eigendecomposition of $A^T A$. So $V$ is the eigenvector matrix of $A^T A$, and the squared singular values $\sigma_i^2$ are its eigenvalues. Similarly, $AA^T = U \Sigma^2 U^T$ gives us $U$.

Remember how we said eigendecomposition fails because it tries to use the same basis for both input and output ($P$ and $P^{-1}$)? SVD fixes this by using two different bases.

1. The Rectangular Fix (Decoupling Dimensions)If $A$ is a $100 \times 3$ matrix, it maps a 3D vector to a 100D vector. Eigendecomposition crashes because you cannot have a single basis that spans both 3D and 100D space simultaneously.SVD says: "Fine. I will use a 3D basis ($V$) for the input and a 100D basis ($U$) for the output."$V^T$ rotates the 3D input to align with the axes.$\Sigma$ scales those 3 axes and "pads" the rest with zeros to jump to 100D.$U$ rotates the result in 100D space. Problem solved. The input and output spaces no longer need to match.

2. The Defective Fix (The Symmetric Guarantee)For a shear matrix where eigenvectors collapse into a single line, eigendecomposition fails because it runs out of independent vectors to form a basis.SVD sidesteps this by looking at $A^T A$ instead of $A$. Even if $A$ is a mess, $A^T A$ is always a symmetric, positive semi-definite matrix.Symmetric matrices are practically the "nicest" matrices in linear algebra. The Spectral Theorem guarantees that they always have a full set of orthogonal eigenvectors. By building our SVD bases ($U$ and $V$) from the eigenvectors of these "nice" symmetric matrices ($AA^T$ and $A^T A$), SVD guarantees that a valid decomposition exists for every matrix, no matter how distorted or defective it is.

However, computing eigenvectors exactly is expensive. So how do we actually do it? We will discuss a simple numerical maethod to dot. But most modern libraries like NumPy use more sophisticated methods.

### The Power Method

SVD basically gives us the directions of maximum, second maximum, third maximum... scaling of $A^T A$. Why? Because when we rotate a vector with $V^T$, we are moving into the eigenbasis of $A^T A$. There, $\Sigma$ will stretch the most along the eigenvector with the largest eigenvalue, less along the second largest, and so on.

Using this intuition: if we keep applying the transformation $A^T A$ to a random vector, the component along the largest eigenvector will become so large that we can ignore the others. The vector converges to point almost entirely in the dominant eigenvector direction. Deflate that direction out and repeat for the next one. This is the **power method** — and it is how SVD is computed in practice without explicitly solving the characteristic polynomial. I have not discussed this section in much detail as I feel that we are not interested in doing calculation by hand to calculate eigenvectors in this blog.

## 5  From SVD to PCA — The Covariance Connection

We understand that SVD gives us the directions of maximum stretching in an orderly fashion. Now, if we think about this in a real-world scenario: say we have data in 100 dimensions. Analysing in 100 dimensions will be tough. However, with the help of SVD, we can take 3 or 10 dimensions which scale our data the most and do our analysis there. That is the core idea of PCA.

But let us make the connection precise.

### The Covariance Matrix

Given a centered data matrix $X$ ($n$ samples × $p$ features), the sample covariance matrix is:

$$
S = \frac{X^T X}{n - 1}
$$

$S$ is a $p \times p$ symmetric matrix. Its eigenvectors are the directions of maximum variance, and its eigenvalues tell us how much variance each direction captures. PCA asks us to find these eigenvectors.

### SVD Gives Us Covariance Eigenvectors for Free

Take the SVD of the centered data: $X = U \Sigma V^T$. Now compute the covariance:

$$
S = \frac{X^T X}{n-1} = \frac{(V \Sigma^T U^T)(U \Sigma V^T)}{n-1} = V \left(\frac{\Sigma^2}{n-1}\right) V^T
$$

This is already the eigendecomposition of $S$! The columns of $V$ are the eigenvectors of the covariance matrix, called the **principal components**. And $\sigma_i^2 / (n-1)$ are the eigenvalues, the **variance explained** by each component.

We never need to form the covariance matrix explicitly. SVD of the data hands us the answer directly.

> **PCA is not a separate algorithm with its own theory.** It is SVD applied to a centered data matrix. The principal components are the right singular vectors ($V$), and the variance explained is encoded in the singular values ($\Sigma$).

To reduce from $p$ dimensions to $k$, keep the first $k$ columns of $V$ and project: $X_{\text{reduced}} = X V_k$. This gives us an $n \times k$ matrix — our data expressed in the $k$ most informative directions.

We will discuss Covariance as a concept and what it does, how is it useful in more detail when we move on to probability.

### PCA in Code

Here is a minimal implementation showing how SVD gives us PCA, and verifying that it matches the eigendecomposition of the covariance matrix.

```python
import numpy as np

# Generate some correlated 2D data
np.random.seed(42)
X = np.random.randn(200, 2) @ np.array([[2, 1], [1, 3]])

# Step 1: Center the data (subtract the mean of each feature)
X_centered = X - X.mean(axis=0)

# Step 2: SVD of the centered data matrix
U, sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)

# The principal components are the rows of Vt (columns of V)
V = Vt.T
print("Principal components (columns of V):")
print(V)

# Variance explained by each component
variance_explained = (sigma ** 2) / (len(X) - 1)
print("\nVariance explained:", variance_explained)
print("Ratio:", variance_explained / variance_explained.sum())

# Step 3: Project data onto first k=1 principal component
k = 1
X_reduced = X_centered @ V[:, :k]
print(f"\nOriginal shape: {X_centered.shape}")
print(f"Reduced shape:  {X_reduced.shape}")

# --- Verify: this matches the covariance eigenvectors ---
S = (X_centered.T @ X_centered) / (len(X) - 1)
eigenvalues, eigenvectors = np.linalg.eigh(S)

# eigh returns ascending order; flip to match SVD's descending
eigenvalues = eigenvalues[::-1]
eigenvectors = eigenvectors[:, ::-1]

print("\n--- Verification ---")
print("SVD variance:       ", variance_explained)
print("Covariance eigenval:", eigenvalues)
print("These should match (and they do).")
```

## 6  Tying It All Together

Let us trace the thread. We started with **change of basis**,i.e., the idea that the same point or transformation can look simpler from a different viewpoint. We saw a point $[6, 2]^T$ become $[4, 2]^T$ just by switching rulers, and learnt that $T_{\text{new}} = C^{-1}TC$ lets us rewrite any transformation in any basis.

That led us to **eigenvectors**: directions where $T$ only scales, never rotates. If we build a basis entirely out of eigenvectors, $T$ becomes a diagonal matrix of eigenvalues. We used this to turn $T^{100}$ from a nightmare into a one-liner.

Then we moved to **decomposition** — factorising matrices into simpler pieces. **Eigendecomposition** ($A = P\Lambda P^{-1}$) breaks a square matrix into its eigenvector and eigenvalue components. But it only works for square, diagonalisable matrices. **SVD** ($A = U\Sigma V^T$) generalises this to *any* matrix by decomposing it into rotation + scale + rotation, revealing the directions of maximum stretching.

Finally, we connected SVD to **PCA**: the covariance matrix $S = X^TX/(n-1) = V(\Sigma^2/(n-1))V^T$ shows that SVD of the data *is* the eigendecomposition of the covariance. The right singular vectors are the principal components, the singular values encode the variance, and dimensionality reduction is just keeping the top $k$ directions.

Each concept was a stepping stone to the next.

---

## References

1. **Deisenroth, M. P., Faisal, A. A., & Ong, C. S.** (2020). *Mathematics for Machine Learning*. Cambridge University Press. — Chapters 2–4 cover linear algebra, eigendecomposition, and SVD with a machine learning perspective. Freely available at [mml-book.github.io](https://mml-book.github.io/).

2. **Strang, G.** (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press. — Chapters 6 and 7 provide thorough treatments of eigenvalues, SVD, and PCA with geometric intuition.

3. **3Blue1Brown** (2016). *Essence of Linear Algebra* [Video series]. YouTube. — Exceptional visual explanations of change of basis (Chapter 13), eigenvectors (Chapter 14), and abstract vector spaces. Several intuitions in this post — particularly the idea that eigenvectors are directions where a transformation "collapses to pure scaling," and the rotation + scale + rotation geometry of SVD — are heavily influenced by Grant Sanderson's visual presentations in this series. Available at [3blue1brown.com/topics/linear-algebra](https://www.3blue1brown.com/topics/linear-algebra).

4. **Shlens, J.** (2014). *A Tutorial on Principal Component Analysis*. arXiv:1404.1100. — A concise derivation of PCA from the covariance matrix and its connection to SVD. Available at [arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100).

5. **Golub, G. H., & Van Loan, C. F.** (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press. — The definitive reference on SVD algorithms, the power method, and numerical linear algebra.

6. **Wall, M. E., Rechtsteiner, A., & Rocha, L. M.** (2003). Singular Value Decomposition and Principal Component Analysis. In *A Practical Approach to Microarray Data Analysis* (pp. 91–109). Springer. — A practical walkthrough of SVD-based PCA with worked examples.

---

*This is Part 1 of a series on the mathematical foundations of deep learning. Part 2 will cover multivariate calculus which will help in building a simple neural network from scratch.*
