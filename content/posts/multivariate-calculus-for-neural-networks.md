---
title: "Multivariate Calculus for Neural Networks"
date: 2025-02-13
description: "From partial derivatives to gradient descent — the calculus you need to train a neural network from scratch."
tags: ["calculus", "machine-learning", "mathematics"]
series: ["Math for ML"]
draft: false
---

# Multivariate Calculus for Neural Networks

*From partial derivatives to gradient descent. The calculus you need to train a neural network from scratch.*

---

## 1  Why Calculus?

In the [last post](/blog/linear-algebra-for-pca), we built up linear algebra from change of basis all the way to PCA. We learnt how to manipulate vectors using linear transformations: rotate them, scale them, reflect them, shear them.

But here is the problem: no matter how many linear transformations we stack, we can never *curve* or *bend* a vector's path. Linear transformations are, well, linear. A straight line goes in, a straight line comes out. Always.

And the real world is full of curves. The boundary that separates cats from dogs in an image is not a straight line. The relationship between a drug's dosage and its effect is not linear. The function that maps millions of pixel values to the word "cat" is spectacularly nonlinear.

So if we want to *approximate* curvy things (and ultimately, that is what a neural network does), we need a way to talk about *change*. How does the output change when we nudge the input a little? If we nudge in *this* direction versus *that* direction, which one reduces the error faster? Calculus gives us the language and the tools to answer these questions precisely.

Here is our target for this post: we will build up multivariate calculus, piece by piece, until we arrive at **gradient descent**, the algorithm that powers the training of almost every neural network in existence. Along the way, we will not skip the subtle logical bridges that most treatments gloss over. If there is a "why" between two concepts, we will stop and answer it, even if it seems trivial. In the next post, we will actually use all of this to train a basic neural network that predicts an "X" on a 3×3 screen.

Let us start with a quick recap and then build from there.

---

## 2  A Quick Calculus Recap

I will assume you have seen basic differential calculus before. But here is a rapid refresher of the rules we will lean on. If these look familiar, skim through. If not, spend a minute with each, as they are the atoms everything else is built from.

**Power Rule:**

$$
f(x) = x^n \implies f'(x) = n \cdot x^{n-1}
$$

Straightforward. The exponent comes down as a coefficient, and the exponent drops by one.

**Product Rule:**

$$
\frac{d}{dx}\big[f(x) \cdot g(x)\big] = f'(x) \cdot g(x) + f(x) \cdot g'(x)
$$

When two functions are multiplied, the derivative is: "differentiate the first and keep the second, plus keep the first and differentiate the second." Each function gets its turn while the other stays put.

**Chain Rule** (single variable):

$$
\frac{d}{dx} f\big(g(x)\big) = f'\big(g(x)\big) \cdot g'(x)
$$

This one is the most important for us. In plain words: if the output depends on an intermediate variable, and that intermediate variable depends on the input, then the total rate of change is the *product* of the two individual rates. Differentiate the outer function (evaluated at the inner function), then multiply by the derivative of the inner function.

Think of it like a relay race. If runner A passes the baton to runner B, the total speed of the baton through the relay is not either runner's speed alone; it is how they *compose*. The chain rule captures this composition.

We will generalise this to multiple variables shortly, and that generalisation is what makes backpropagation, and therefore all of deep learning, work.

---

## 3  Partial Derivatives

So far, all our functions had one input and one output. $f(x) = x^2$, $f(x) = \sin(x)$, and so on. One knob, one dial. Twist the knob, the dial moves.

But real problems are rarely that simple. A neural network's loss function might depend on *millions* of weights simultaneously. Even a simple function like the area of a rectangle, $A = l \times w$, depends on two variables, length and width. We need a way to differentiate when there are multiple inputs.

This is where partial derivatives come in, and the idea is genuinely simple: **differentiate with respect to one variable, and pretend everything else is a constant.**

That is the whole trick. You already know how to differentiate single-variable functions. A partial derivative just says: "focus on one variable, and as far as the differentiation is concerned, all the other variables are just numbers, constants that happen to have letter names."

Let us do a quick example. Say $f(x, y) = x^2 y$.

To find $\frac{\partial f}{\partial x}$: treat $y$ as a constant. So $f = (y) \cdot x^2$, and differentiating with respect to $x$ gives:

$$
\frac{\partial f}{\partial x} = 2xy
$$

To find $\frac{\partial f}{\partial y}$: treat $x$ as a constant. So $f = (x^2) \cdot y$, and differentiating with respect to $y$ gives:

$$
\frac{\partial f}{\partial y} = x^2
$$

That is all there is to it. No new rules, no new mechanics. The notation $\frac{\partial}{\partial x}$ just means "differentiate with respect to $x$, everything else is frozen." Once you internalise this, partial derivatives are as mechanical as regular ones.

The only new thing is an *interpretation*: $\frac{\partial f}{\partial x}$ tells you how sensitive the output is to changes in $x$ *while holding $y$ fixed*. Imagine you are adjusting two knobs on a radio, one for volume and one for bass. The partial derivative with respect to volume tells you how the sound changes as you turn the volume knob, *keeping the bass knob perfectly still*. Each partial derivative isolates the effect of one knob.

---

## 4  The Multivariable Chain Rule

Now here is where things get genuinely interesting, and where we start building toward backpropagation.

Suppose we have a function $z = f(x, y)$, but $x$ and $y$ are not free variables. They themselves depend on some other variables $u$ and $v$:

$$
x = x(u, v) \qquad y = y(u, v)
$$

Now I ask: how does $z$ change when we nudge $u$?

This is not as straightforward as before. Changing $u$ does not directly appear in $f$, but it *does* affect $x$, and $x$ appears in $f$. And changing $u$ also affects $y$, and $y$ also appears in $f$. So there are *two paths* through which $u$ influences $z$: one through $x$, and one through $y$. Both contribute, and we need to sum them up.

$$
\frac{\partial z}{\partial u} = \frac{\partial z}{\partial x} \cdot \frac{\partial x}{\partial u} + \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial u}
$$

And by the exact same logic for $v$:

$$
\frac{\partial z}{\partial v} = \frac{\partial z}{\partial x} \cdot \frac{\partial x}{\partial v} + \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial v}
$$

Think of it like water flowing downhill through a network of streams. The water (the change) starts at $u$ and needs to reach $z$. It can flow through the $x$-stream or the $y$-stream (or both). Each stream has two factors: how much water gets from $u$ to that intermediate variable, and how much that intermediate variable affects $z$. The total flow is the sum of all streams.

Each term is: *"how much does $z$ care about this intermediate variable"* times *"how much does that intermediate variable change when I poke $u$."* Every path from input to output contributes.

### Deriving It Properly

Let us not just state this. Let us derive it from scratch, so you can see there is no magic.

We start from the **total differential**. If $z = f(x, y)$, and we make small changes $dx$ and $dy$ to the inputs, the resulting change in $z$ is:

$$
dz = \frac{\partial f}{\partial x}\, dx + \frac{\partial f}{\partial y}\, dy
$$

Where does this come from? It is the linear approximation idea applied to a small neighbourhood. The total change in $z$ equals "how much $z$ changes per unit change in $x$, times the actual change in $x$," plus the same for $y$. It is the multivariable version of the familiar $df = f'(x)\, dx$.

Now, $x$ and $y$ themselves depend on $u$ and $v$. So their changes can be broken down further:

$$
dx = \frac{\partial x}{\partial u}\, du + \frac{\partial x}{\partial v}\, dv
$$

$$
dy = \frac{\partial y}{\partial u}\, du + \frac{\partial y}{\partial v}\, dv
$$

The same idea again: $x$ depends on two things, so its total change has two parts.

Substituting these into the expression for $dz$:

$$
dz = \frac{\partial f}{\partial x}\left(\frac{\partial x}{\partial u}\, du + \frac{\partial x}{\partial v}\, dv\right) + \frac{\partial f}{\partial y}\left(\frac{\partial y}{\partial u}\, du + \frac{\partial y}{\partial v}\, dv\right)
$$

Now expand and collect terms by $du$ and $dv$:

$$
dz = \underbrace{\left(\frac{\partial f}{\partial x}\frac{\partial x}{\partial u} + \frac{\partial f}{\partial y}\frac{\partial y}{\partial u}\right)}_{\partial z / \partial u}\, du \ +\  \underbrace{\left(\frac{\partial f}{\partial x}\frac{\partial x}{\partial v} + \frac{\partial f}{\partial y}\frac{\partial y}{\partial v}\right)}_{\partial z / \partial v}\, dv
$$

But we also know from the total differential of $z$ with respect to $u$ and $v$ that:

$$
dz = \frac{\partial z}{\partial u}\, du + \frac{\partial z}{\partial v}\, dv
$$

Matching the coefficients of $du$ and $dv$ gives us exactly the chain rule formulas above. Nothing is assumed, no hand-waving. It falls out from substitution and collecting like terms.

### The General Form

This generalises naturally. If $z = f(x_1, x_2, \ldots, x_k)$ and each $x_i$ depends on variables $u_1, u_2, \ldots, u_p$, then:

$$
\frac{\partial z}{\partial u_j} = \sum_{i=1}^{k} \frac{\partial z}{\partial x_i} \cdot \frac{\partial x_i}{\partial u_j}
$$

Every path from $u_j$ to $z$ through any intermediate variable $x_i$ gets its contribution summed up. This is the core mathematical idea behind **backpropagation** [6]: computing how the loss changes with respect to every weight by tracing the chain of dependencies backwards through the network. But we are getting ahead of ourselves. One step at a time.

---

## 5  From Scalar Functions to Vector Functions: The Jacobian

So far, our function $f$ has been producing a single output, a scalar. We differentiated scalar functions of one variable, then scalar functions of multiple variables, and we chained them. Good.

But what happens when a function takes $n$ inputs and produces $m$ outputs? That is, $f : \mathbb{R}^n \to \mathbb{R}^m$. The function eats a vector and spits out a different vector.

This is not exotic at all. A single layer of a neural network does exactly this. It takes a vector of activations (say, 128 numbers) and produces a new vector of activations (say, 64 numbers). If we want to understand how the layer's outputs change when we wiggle its inputs, we need a way to capture *all* those sensitivities at once.

### Linear Approximation Motivates the Jacobian

Let us build this up from what we already know. For a single-variable function, the linear approximation says:

$$
f(u + h) \approx f(u) + f'(u) \cdot h
$$

The derivative $f'(u)$ is a single number: it tells us the rate. If we step by a small amount $h$, the output changes by approximately $f'(u) \cdot h$. Simple.

Now we want the same thing, but for a vector function. We want some kind of "derivative object" that tells us: if I nudge the input vector by a small vector $\mathbf{h}$, how does the *entire output vector* respond?

$$
f(\mathbf{u} + \mathbf{h}) \approx f(\mathbf{u}) + \ ??? \cdot \mathbf{h}
$$

What should "???" be? Let us think about this carefully. The input nudge $\mathbf{h}$ is a vector in $\mathbb{R}^n$. The change in output needs to be a vector in $\mathbb{R}^m$. So "???" must be something that takes vectors in $\mathbb{R}^n$ and produces vectors in $\mathbb{R}^m$. That is a **linear map**, and a linear map from $\mathbb{R}^n$ to $\mathbb{R}^m$ is represented by an $m \times n$ matrix.

This matrix is the **Jacobian**:

$$
f(\mathbf{u} + \mathbf{h}) \approx f(\mathbf{u}) + \mathbf{J} \cdot \mathbf{h}
$$

### What Is Inside the Jacobian?

The Jacobian is all the partial derivatives, arranged in a grid:

$$
\mathbf{J} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\[6pt]
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\[6pt]
\vdots & \vdots & \ddots & \vdots \\[6pt]
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

Do not let the size intimidate you. Look at it this way:

**Each row $i$** contains all the partial derivatives of the $i$-th output $f_i$. It tells you: "how does output $i$ respond when I wiggle each of the $n$ inputs, one at a time?" It is the sensitivity profile of that single output, its personal row of knobs. If you care only about how $f_3$ changes, you look at row 3 and nothing else.

**Each column $j$** contains the partial derivative of every output with respect to input $j$. It tells you: "if I nudge input $j$ by a tiny amount, how do *all* $m$ outputs react?" It is the ripple effect of poking one input, and every output that cares about input $j$ shows up in this column.

So the Jacobian is really just the total derivative, organised into a matrix for clean notation and clear computation. There is nothing conceptually new beyond partial derivatives. It is bookkeeping. A very elegant and useful piece of bookkeeping.

And here is the satisfying part: if you actually carry out the matrix-vector product $\mathbf{J} \cdot \mathbf{h}$, the $i$-th component of the result is:

$$
(\mathbf{J} \cdot \mathbf{h})_i = \sum_{j=1}^{n} \frac{\partial f_i}{\partial x_j} h_j
$$

That is exactly the multivariable chain rule formula from Section 4, the sum over all paths from input changes to output $i$. The Jacobian does not introduce any new mathematics. It just packages the chain rule into a matrix multiply so that we can compute all the output changes simultaneously with one operation.

---

## 6  The Gradient: A Special Case of the Jacobian

Here is where we need to be careful, and where this blog diverges from most treatments.

Most textbooks go straight from the Jacobian to the gradient. They say "the gradient is the vector of partial derivatives" and move on. Students nod, write the formula, and use it. But there is a subtle and important distinction between the Jacobian and the gradient that, if skipped, leaves a gap in your understanding. It is one of those things that seems trivial on the surface ("they have the same entries, what is the big deal?"), but the "big deal" is *where these objects live* and *what we can do with them*.

This blog does not skip logical bridges, even seemingly trivial ones. So let us cross this one properly.

### Why Do We Care About a Special Case?

In any optimisation problem (and training a neural network *is* an optimisation problem), we ultimately care about a single number: the **loss** (or error). No matter how complex the network, no matter how many layers or millions of parameters, after the forward pass we compress everything into one scalar: "your prediction was this far from the truth." One number.

So the function we care about optimising is $f : \mathbb{R}^n \to \mathbb{R}$. It takes $n$ inputs (all the weights of the network) and produces 1 output (the loss).

Now, what does the Jacobian look like for this function? The Jacobian is an $m \times n$ matrix, and here $m = 1$. So it collapses to a **$1 \times n$ row vector**:

$$
\mathbf{J} = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \frac{\partial f}{\partial x_2} & \cdots & \frac{\partial f}{\partial x_n} \end{bmatrix}
$$

And the linear approximation becomes:

$$
f(\mathbf{x} + \mathbf{h}) \approx f(\mathbf{x}) + \mathbf{J} \cdot \mathbf{h} = f(\mathbf{x}) + \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} h_i
$$

This works perfectly well. We can plug in any direction $\mathbf{h}$ and get the approximate change in $f$. But there is a subtle problem lurking here.

### The Dual Space Problem

The Jacobian $\mathbf{J}$ is a $1 \times n$ **row vector**. The direction $\mathbf{h}$ we want to move in is an $n \times 1$ **column vector**. They have the same number of entries, but they are not the same kind of object.

Here is what that means concretely. The Jacobian is a *linear functional*. It is a machine that *eats* a vector and *spits out* a number. Mathematically, it lives in the **dual space** $(\mathbb{R}^n)^*$, not in $\mathbb{R}^n$ itself. Meanwhile, the direction $\mathbf{h}$ that we want to walk in lives in $\mathbb{R}^n$: it is a point in the original vector space, with a length and a direction you can visualise.

So the Jacobian gives us the right *value* for our approximation, no dispute there. But it does not directly help us answer the question we actually care about: "which *direction* should I move to reduce the loss the fastest?" Because "directions" are vectors in $\mathbb{R}^n$, and the Jacobian is not a vector in $\mathbb{R}^n$. They live in different spaces. We cannot directly compare them, point at the Jacobian and say "walk that way."

Let me make this concrete with an analogy.

**Imagine you are standing on a hillside.** The **Jacobian** is like a machine sitting next to you. You tell it, *"I am thinking of walking northeast,"* and it replies, *"you will gain 3 metres of elevation."* You ask, *"what about due south?"* and it replies, *"you will lose 1.5 metres."* It answers *queries* about specific directions. It is a function that maps directions to elevation changes. It is incredibly useful. But you cannot point at the machine and say "walk toward the machine." The machine is not a direction. It does not live on the hillside with you. It is a *response oracle* that evaluates directions, not a direction itself.

What you actually want is an **arrow painted on the ground** pointing uphill, something that *is* a direction, that lives on the hillside with you, that you can walk along, measure angles against, or follow. You could derive that arrow from the machine: ask the machine about every possible direction, find which one it rates highest, and paint that direction on the ground. But the arrow and the machine are different representations of the same underlying information. The machine is the Jacobian. The arrow is the **gradient**.

### From Jacobian to Gradient

The fix is elegant. Instead of the matrix-vector product $\mathbf{J} \cdot \mathbf{h}$ (row vector times column vector), we write the same calculation as an **inner product** between two column vectors:

$$
f(\mathbf{x} + \mathbf{h}) \approx f(\mathbf{x}) + \langle \nabla f, \mathbf{h} \rangle
$$

where $\nabla f$ (the **gradient**) is the column vector version of the Jacobian:

$$
\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}
$$

**Numerically, nothing changed.** The inner product $\langle \nabla f, \mathbf{h} \rangle$ produces the exact same number as $\mathbf{J} \cdot \mathbf{h}$. You can verify: both give $\sum_i \frac{\partial f}{\partial x_i} h_i$. Same entries, same sum, same result.

**Conceptually, everything changed.** The gradient $\nabla f$ is now a column vector in $\mathbb{R}^n$, the *same space* as $\mathbf{h}$. They are the same kind of object. We can compare them. We can compute the angle between them. We can ask: "is $\mathbf{h}$ aligned with $\nabla f$, or perpendicular to it, or pointing the opposite way?" None of these questions made sense when one object was a row vector in the dual space and the other was a column vector in the original space.

Going back to the hillside: we derived the arrow from the machine. The arrow (gradient) and the machine (Jacobian) encode the same information about how the elevation changes. But the arrow lives on the hillside with you, in your space, and you can directly use it to decide where to walk.

This distinction between the Jacobian as a dual-space machine and the gradient as a primal-space arrow is one of those subtle things that most courses skip because "they give the same answer anyway." And they do [1]. But if you skip the bridge, you never understand *why* the gradient is defined as a column vector, or *why* we use the inner product instead of matrix multiplication. It looks like an arbitrary notational preference when it is actually a meaningful geometric shift: we moved from "a function that evaluates directions" to "a direction itself."

### The Direction of Steepest Ascent

Now here is the payoff of putting the gradient in the same space as $\mathbf{h}$. Since both are vectors in $\mathbb{R}^n$, their inner product has a geometric interpretation:

$$
\langle \nabla f, \mathbf{h} \rangle = \|\nabla f\| \cdot \|\mathbf{h}\| \cdot \cos\theta
$$

where $\theta$ is the angle between $\nabla f$ and $\mathbf{h}$.

This means the approximate change in $f$ when we step by $\mathbf{h}$ is:

$$
\Delta f \approx \|\nabla f\| \cdot \|\mathbf{h}\| \cdot \cos\theta
$$

Now ask: for a fixed step size $\|\mathbf{h}\|$, when is this change **maximised**?

$\|\nabla f\|$ is fixed (it depends on where we are, not where we are going). $\|\mathbf{h}\|$ is fixed (we chose our step size). The only thing we control is $\theta$, the angle between our step direction and the gradient.

$\cos\theta$ is maximised when $\theta = 0$, giving $\cos 0 = 1$. That is, $\mathbf{h}$ should point in the **same direction** as $\nabla f$.

So the gradient points in the direction of steepest ascent. No hand-waving, no appeals to geometric intuition without proof. It falls straight out of the inner product formula and the properties of cosine. If you want to increase $f$ as fast as possible for a given step size, walk in the direction of $\nabla f$.

And the magnitude $\|\nabla f\|$ tells you *how steep* that steepest ascent is. A large gradient means the hill is steep; a small gradient means it is nearly flat. When $\nabla f = \mathbf{0}$, the hill is perfectly flat in every direction, and you are at a critical point (a minimum, maximum, or saddle point).

---

## 7  Taylor Series: Better Approximations

We have been using *linear* approximation this whole time: approximate the function by its value plus a first-order correction. This is great when the step is tiny, but the approximation degrades as the step gets larger. We are fitting a tangent line (or tangent plane) to a curve (or surface), which is good locally, but the curve bends away.

What if we want to capture some of that bending? What if we want to know not just the slope, but the *curvature*? That is where the Taylor series comes in. It gives us a systematic way to build better and better approximations by including higher-order derivative information.

### Single Variable First

For a well-behaved function (continuous, $n$-times differentiable), the Taylor series expansion around a point $p$ is:

$$
f(x) = \sum_{k=0}^{\infty} \frac{f^{(k)}(p)}{k!}(x - p)^k
$$

Expanding the first few terms:

$$
f(x) = f(p) + f'(p)(x-p) + \frac{f''(p)}{2!}(x-p)^2 + \frac{f'''(p)}{3!}(x-p)^3 + \cdots
$$

When $p = 0$, this is called the **Maclaurin series**.

Let us use a cleaner notation for what follows. Write $x = p + \delta$ where $\delta$ is a small step away from $p$:

$$
f(p + \delta) = f(p) + f'(p)\,\delta + \frac{f''(p)}{2}\,\delta^2 + \frac{f'''(p)}{6}\,\delta^3 + \cdots
$$

Each successive term adds a higher-order correction. Think of it as zooming in on the function with increasingly powerful lenses:

- The **zeroth-order** term $f(p)$ says: "the function is approximately constant near $p$." This is a flat line. Terrible approximation for anything non-trivial, but technically valid for an infinitely small neighbourhood.
- Add the **first-order** term $f'(p)\delta$ and you get the tangent line, the **linearisation**. This captures the slope.
- Add the **second-order** term $\frac{f''(p)}{2}\delta^2$ and you get a parabola that hugs the curve more tightly. This captures the curvature, specifically whether the function is bending up or down.
- Each additional term bends the approximation a little more to match the true function.

**A note on terminology** that trips people up: if we keep only the first two terms, $f(p) + f'(p)\delta$, we call this a *first-order* Taylor expansion because we used up to the first derivative. The *error* we incur by truncating here is of order $O(\delta^2)$, meaning it scales as $\delta^2$ for small $\delta$. Some people say this is "second-order accurate" because the error is second-order, but that can be confused with "we used the second derivative," which we did not. To be precise: we have a first-order expansion, and the error is second-order. Keep these two notions separate, as it avoids confusion when you encounter phrases like "second-order method" later (which genuinely means using $f''$).

### Multivariate Taylor Series

The same idea extends to functions of multiple variables. For $f(x, y)$ near a point $(x, y)$, stepping by $(\delta x, \delta y)$:

$$
\begin{aligned}
f(x + \delta x,\ y + \delta y) &= f(x, y) + \left(\frac{\partial f}{\partial x}\,\delta x + \frac{\partial f}{\partial y}\,\delta y\right) \\
&\quad + \frac{1}{2}\left(\frac{\partial^2 f}{\partial x^2}(\delta x)^2 + 2\frac{\partial^2 f}{\partial x \partial y}\,\delta x\,\delta y + \frac{\partial^2 f}{\partial y^2}(\delta y)^2\right) + \cdots
\end{aligned}
$$

The **first-order terms** are the gradient dotted with the step, which we already know. It captures the slope in the direction of $\boldsymbol{\delta}$.

The **second-order terms** capture the curvature, meaning how the slope itself is changing. Notice there are *three* second-order terms for two variables: $\frac{\partial^2 f}{\partial x^2}$ tells us how the slope in $x$ changes as we move in $x$; $\frac{\partial^2 f}{\partial y^2}$ tells us how the slope in $y$ changes as we move in $y$; and $\frac{\partial^2 f}{\partial x \partial y}$ is the *cross-term*, capturing how the slope in $x$ changes as we move in $y$ (or equivalently, how the slope in $y$ changes as we move in $x$). The factor of 2 in front of the cross-term is not a typo. It comes from the binomial expansion of $(\delta x \frac{\partial}{\partial x} + \delta y \frac{\partial}{\partial y})^2$.

We can write all of this compactly using vectors and matrices. Let $\boldsymbol{\delta} = [\delta x,\ \delta y]^T$:

$$
f(\mathbf{x} + \boldsymbol{\delta}) \approx f(\mathbf{x}) + \nabla f^T \boldsymbol{\delta} + \frac{1}{2} \boldsymbol{\delta}^T \mathbf{H} \boldsymbol{\delta}
$$

where $\nabla f$ is the gradient (all the first-order information packed into a vector) and $\mathbf{H}$ is the **Hessian matrix** (all the second-order information packed into a matrix). We will study the Hessian in detail later in this post. For now, just notice the elegant structure: gradient captures slope, Hessian captures curvature, and higher-order terms (which we truncated) capture increasingly fine-grained shape details of the function.

---

## 8  Gradient Descent

Now we arrive at the main event, the algorithm that actually trains neural networks. Gradient descent was first proposed by Cauchy in 1847 [10], and it remains the foundation of all modern optimisation in machine learning.

We have a loss function $f(\mathbf{x})$ that depends on our model's parameters $\mathbf{x}$ (all the weights and biases). We want to *minimise* it, to make the model as accurate as possible.

From Section 6, we know the gradient $\nabla f$ points in the direction of steepest **ascent**. So if we want to go **down**, that is, reduce the loss, we simply go in the *opposite* direction.

From our linear approximation:

$$
f(\mathbf{x} + \boldsymbol{\delta}) \approx f(\mathbf{x}) + \langle \nabla f, \boldsymbol{\delta} \rangle
$$

Now, let us choose our step to be $\boldsymbol{\delta} = -\alpha \nabla f$, meaning we step in the *negative* gradient direction, scaled by some factor $\alpha$. Substituting:

$$
f(\mathbf{x} - \alpha \nabla f) \approx f(\mathbf{x}) + \langle \nabla f,\ -\alpha \nabla f \rangle
$$

The inner product is linear, so we pull out $-\alpha$:

$$
= f(\mathbf{x}) - \alpha \langle \nabla f, \nabla f \rangle
$$

And the inner product of a vector with itself is the squared norm:

$$
= f(\mathbf{x}) - \alpha \|\nabla f\|^2
$$

Two things guarantee this works:

1. **The squared norm $\|\nabla f\|^2$ is always non-negative.** It is a sum of squares, so it can never be negative. So $-\alpha\|\nabla f\|^2$ is always non-positive (assuming $\alpha > 0$). This means $f(\mathbf{x} - \alpha \nabla f) \leq f(\mathbf{x})$. The function value *decreases*. We are guaranteed to go downhill. (The only exception is when $\nabla f = \mathbf{0}$, in which case $\|\nabla f\|^2 = 0$ and we do not move, but that means we are already at a critical point, so there is no downhill direction anyway.)

2. **We already proved the gradient is the direction of steepest ascent.** So its negative is the direction of **steepest descent**. We are not just going downhill. We are going downhill *as fast as possible* for the given step size.

The scalar $\alpha$ is called the **learning rate**. It controls how large of a step we take. The update rule for gradient descent is:

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k)
$$

Repeat this over and over: compute the gradient at the current position, take a step in the opposite direction, arrive at a new position, compute the gradient again, step again. The parameters slowly (or quickly, depending on $\alpha$) crawl toward a minimum of the loss function. That is gradient descent, the workhorse of neural network training.

### Why Step Size Matters

Here is the catch. Our entire derivation relied on the linear approximation, which dropped the higher-order Taylor terms. That approximation is only accurate when $\boldsymbol{\delta}$ is *small*. If we take a huge step, the quadratic and higher terms that we ignored are no longer negligible, and our linear prediction ("the function will decrease by $\alpha\|\nabla f\|^2$") might be completely wrong. The function could actually *increase*.

Imagine walking along a narrow mountain ravine. The gradient says "go left, that is the steepest descent." And it is right, locally, at the exact spot where you are standing, going left is the fastest way down. But if you take an enormous leap to the left, you fly right past the bottom of the ravine and land high up on the opposite wall, *higher* than where you started. The gradient was correct about the *direction*, but the distance was too large for the linear approximation to hold.

> Think of it like using Google Maps in a winding alley. The blue arrow says "go straight." And right now, that is correct. But if you walk 500 metres straight without rechecking, you will probably walk through a building. The arrow was right *locally*, but the path curves. You need to take small steps and recalculate the direction after each one.

So the learning rate $\alpha$ needs to be chosen carefully. Too large, and you overshoot: the loss *increases* instead of decreasing, and training diverges. Too small, and each step is so tiny that training takes an eternity to converge. The right balance depends on the geometry of the loss landscape.

But how do we know what "too large" means? Is there a principled, mathematical way to choose $\alpha$?

Yes, there is. And it comes from formalising exactly how fast the gradient is allowed to change.

---

## 9  Lipschitz Continuity: How to Choose Your Step Size

The reason a large step can overshoot is that **the gradient changes as we move**. At our current position, the gradient says "go left." But by the time we arrive at the new position (after a big leftward step), the gradient *there* might say "go right," meaning we have already passed the minimum and gone too far. The faster the gradient changes between positions, the more dangerous a large step becomes.

So the question becomes: **how fast can the gradient change?** If we can put a bound on that, we can bound the step size to stay safe.

### The Idea

A function $g$ is called **Lipschitz continuous** with constant $L$ if, for all $\mathbf{x}$ and $\mathbf{y}$:

$$
\|g(\mathbf{x}) - g(\mathbf{y})\| \leq L \|\mathbf{x} - \mathbf{y}\|
$$

In plain language: the change in the output is bounded by $L$ times the change in the input. No matter where you are, no matter which direction you move, the function cannot "jump" faster than a speed of $L$. The constant $L$ acts as a universal speed limit on the function.

A small $L$ means the function changes slowly and is smooth and predictable. A large $L$ means the function can swing wildly over short distances.

> Think of it as a leash on the function. A dog on a 3-metre leash can move at most 3 metres away from you for every metre you walk. Similarly, a function with Lipschitz constant $L$ can change its output by at most $L \cdot d$ when you change the input by $d$. The leash prevents the function from doing anything too wild.

### Applying It to the Gradient

We apply this concept not to the loss function $f$ itself, but to its **gradient** $\nabla f$. We say the gradient is Lipschitz continuous with constant $L$ if:

$$
\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L \|\mathbf{x} - \mathbf{y}\|
$$

What does this mean? It says: if we move a distance $d$ in parameter space, the gradient vector can change by **at most** $L \cdot d$. The gradient has a speed limit.

A small $L$ means the gradient changes slowly across the landscape. The loss surface is smooth, like gently rolling hills. The slope at one point is a good predictor of the slope at nearby points, so our linear approximation stays valid over larger distances.

A large $L$ means the gradient changes rapidly. The loss surface is rugged, like jagged peaks where the slope can completely reverse direction over a short distance. Our linear approximation degrades quickly, and we can only trust it for very small steps.

### The Step Size Bound

Here is the punchline. Using the Lipschitz condition on the gradient, one can show [4] (via a careful second-order Taylor bound, essentially bounding the error term we dropped in the linearisation) that gradient descent is **guaranteed to decrease the loss at every step** as long as:

$$
\alpha \leq \frac{1}{L}
$$

The classic safe choice is $\alpha = \frac{1}{L}$, which gives the tightest useful guarantee. With this step size, each gradient descent update provably reduces the loss:

$$
f(\mathbf{x}_{k+1}) \leq f(\mathbf{x}_k) - \frac{1}{2L}\|\nabla f(\mathbf{x}_k)\|^2
$$

The intuition is direct: $L$ measures how quickly the slope is changing. If $L$ is large (the gradient changes fast, the terrain is jagged), then $1/L$ is small, so we must take cautious, tiny steps. If $L$ is small (the gradient changes slowly, the terrain is smooth), then $1/L$ is large, and we can stride confidently without fear of overshooting.

> Coming back to the ravine analogy: $L$ tells you how curvy the ravine is. In a gentle, wide valley, you can jog downhill because the slope will not change much between strides. In a narrow, twisting canyon, you better shuffle one foot at a time, because one stride in the "downhill direction" might land you on an uphill wall. The Lipschitz constant quantifies the "twistiness," and $1/L$ tells you the longest safe stride.

### In Practice

In reality, we rarely know $L$ exactly for a neural network's loss landscape. Computing it would require knowing the second derivatives everywhere, which is as expensive (or more) than the problem we are trying to solve. But the theory is not wasted. It is the reason techniques like **learning rate schedules** (start with a bigger $\alpha$, gradually shrink it as training progresses) and **adaptive optimisers** (Adam [3], RMSProp [8], Adagrad [9]) exist.

These methods effectively *estimate the local curvature* on the fly and adjust the step size direction-by-direction. Adam [3], for example, maintains a running estimate of the first and second moments of each gradient component, effectively giving each parameter its own personalised learning rate. The Lipschitz constant is the theoretical foundation behind all of these practical methods [4], and they are all, in some sense, trying to approximate $1/L$ locally without computing it globally.

---

## 10  The Hessian: Capturing Curvature

We briefly met the Hessian in the Taylor series section. Let us now examine it properly, because it answers a question the gradient cannot.

The gradient tells us: *"which direction is downhill, and how steep is it?"* This is valuable, and it is what makes gradient descent work. But the gradient says absolutely nothing about the *shape* of the terrain around us. Is the valley we are descending into wide and gentle, or narrow and steep-sided? Is it even a valley at all, or are we at a saddle point, where the ground curves down in one direction but up in another? The gradient, being a first derivative, is blind to curvature. It only knows the slope at a single point. For curvature, for the shape of the landscape, we need the second derivative.

### The Matrix of Second Derivatives

In single-variable calculus, the second derivative $f''(x)$ captures curvature. In multiple variables, we have many second derivatives, one for each *pair* of input variables, and the Hessian collects all of them into a matrix:

$$
\mathbf{H} = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\[6pt]
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\[6pt]
\vdots & \vdots & \ddots & \vdots \\[6pt]
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}
$$

The **diagonal entries** $\frac{\partial^2 f}{\partial x_i^2}$ tell you how the slope in direction $i$ changes as you move further in direction $i$. This is the curvature along each individual axis, telling you whether the slope is getting steeper or shallower as you go.

The **off-diagonal entries** $\frac{\partial^2 f}{\partial x_i \partial x_j}$ tell you how the slope in direction $i$ changes as you move in direction $j$. These are the cross-curvatures: they capture how the axes interact and influence each other's slopes.

One important structural property: for any reasonably well-behaved function (specifically, when the second partial derivatives are continuous), the order of differentiation does not matter. That is, $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$. This is known as **Schwarz's theorem** (or Clairaut's theorem) [7]. The consequence is that the Hessian is a **symmetric matrix**, and from the last blog post, we know that symmetric matrices have some very nice properties. In particular, they have *real eigenvalues* and *orthogonal eigenvectors*.

This is where things get powerful.

### Connecting to 2D Intuition

Let us first build intuition from single-variable calculus, where curvature analysis is straightforward, and then carefully extend it.

For a function $f(x)$ of one variable, the second derivative test classifies critical points (points where $f'(x) = 0$):

- $f''(x) > 0$: The function is **concave up**, meaning it curves like a bowl $\cup$. The critical point is a **local minimum**. The slope was zero, and as you move away in either direction, the function rises. You are at the bottom.

- $f''(x) < 0$: The function is **concave down**, meaning it curves like a hill $\cap$. The critical point is a **local maximum**. The slope was zero, and as you move away in either direction, the function falls. You are at the top.

- $f''(x) = 0$: **Inconclusive**. The function might have an inflection point (where it switches from curving up to curving down, like a gentle S), or a higher-order extremum. More analysis is needed.

This is clean and complete in 1D because there is only **one direction** to curve in. The second derivative is a single number, and that one number tells the whole curvature story.

But in multiple dimensions, the curvature can be *different in different directions*. A function might curve upward if you walk north, but downward if you walk east. The second derivative in 1D never had to handle this because there was only one direction. In $n$ dimensions, there are infinitely many directions, and the curvature can vary across all of them.

This is exactly what the Hessian and its eigenvalues capture.

### Eigenvalues of the Hessian

The Hessian is a symmetric matrix, so (from our linear algebra blog) it has $n$ real eigenvalues $\lambda_1, \lambda_2, \ldots, \lambda_n$ and $n$ corresponding orthogonal eigenvectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n$.

Each eigenvector defines a **principal direction of curvature**, a direction along which the curvature is "pure," without any cross-coupling. And the corresponding eigenvalue tells you the *amount of curvature* in that direction. Think of each eigenvalue as the $f''(x)$ of a one-dimensional slice through the function along the corresponding eigenvector direction.

With this interpretation, the single-variable rules extend naturally:

**All eigenvalues positive** ($\lambda_i > 0$ for all $i$): The function curves upward in *every* principal direction. No matter which way you move, the surface bends away from you. The shape is bowl-like. In linear algebra language, the Hessian is **positive definite**, meaning the quadratic form $\boldsymbol{\delta}^T \mathbf{H} \boldsymbol{\delta} > 0$ for any nonzero $\boldsymbol{\delta}$. Now, if the gradient is also zero at this point ($\nabla f = \mathbf{0}$), then you are sitting at the bottom of that bowl, a **local minimum**. The positive definite Hessian tells you the *shape* is a bowl; the zero gradient tells you that you are at the *bottom* of it. You need both conditions. A positive definite Hessian halfway up the wall of the bowl still means the shape is bowl-like, but you are not at a minimum there.

**All eigenvalues negative** ($\lambda_i < 0$ for all $i$): The function curves downward in every direction. The shape is hill-like. If the gradient is also zero, you are at the top of that hill, a **local maximum**. The Hessian is **negative definite**.

**Mixed signs** (some positive, some negative): This is the most interesting and, for neural networks, the most common case. The function curves *upward* in some directions and *downward* in others. You are at a **saddle point**.

Picture sitting on a horse saddle. If you look along the horse's spine (say, left-right), the surface curves upward and you are in a valley. If you look perpendicular to the horse (front-back), the surface curves downward and you are on a ridge. At the exact centre of the saddle, the slope is zero in every direction (the gradient is $\mathbf{0}$), so it *looks* like an extremum. But it is neither a minimum nor a maximum. It is a minimum in some directions and a maximum in others. The Hessian is **indefinite**, and the mixed eigenvalue signs reveal this.

**Some eigenvalues zero**: The function is perfectly flat in those directions, with no curvature at all along those eigenvectors. These are degenerate cases. The function might still have a minimum or saddle point, but the second derivative alone cannot distinguish them, and we would need to examine higher-order terms.

> Here is the mental model: in one dimension, $f''(x)$ is a single number, so the concavity has one answer: "up" or "down" or "flat." In $n$ dimensions, the Hessian has $n$ eigenvalues, so the curvature can be independently "up," "down," or "flat" along $n$ orthogonal directions. The eigenvalue decomposition breaks the multi-directional curvature into $n$ independent components, each getting its own "concavity number." It is the 2D second derivative test, generalised dimension by dimension.

### Why This Matters for Neural Network Training

In the loss landscapes of neural networks, which live in spaces with millions or even billions of dimensions, saddle points are **vastly** more common than true local minima [2]. Think about it: for a critical point (where $\nabla f = \mathbf{0}$) to be a genuine local minimum, *every single eigenvalue* of the Hessian must be positive. In a million-dimensional space, that is a million conditions that all need to hold simultaneously. The probability of that, assuming the eigenvalues are somewhat random, is vanishingly small. It is far more likely that some eigenvalues are positive, some are negative, and some are near zero, forming a saddle point.

This is actually good news. At a saddle point, the gradient is zero, so gradient descent stalls momentarily. But in practice, tiny numerical perturbations (floating-point noise, mini-batch randomness in stochastic gradient descent [11]) nudge the parameters slightly off the exact saddle. The negative-eigenvalue directions, which are the "downhill" directions of the saddle, then create a gradient that pulls the optimisation away from the saddle and back on track toward lower-loss regions.

So neural networks rarely get "stuck" at saddle points in practice, even though the landscape is full of them. Understanding the Hessian's eigenvalue structure explains why.

### Connection to Lipschitz Continuity and Step Size

The Hessian also connects directly back to the step size discussion from Section 9. The **largest absolute eigenvalue** of the Hessian, also called the **spectral radius**, $\rho(\mathbf{H}) = \max_i |\lambda_i|$, provides an upper bound on the Lipschitz constant $L$ of the gradient. Why the *absolute* value? Because the Lipschitz constant measures how fast the gradient changes, and the gradient changes rapidly in directions of high curvature regardless of whether that curvature is "up" (positive eigenvalue) or "down" (negative eigenvalue). Consider a saddle point with eigenvalues $\lambda_1 = 2$ and $\lambda_2 = -1000$. The gradient changes far faster along the second direction. If we naively used $1/\lambda_{\max} = 1/2$ as our step size, we would diverge violently along the negative curvature direction. The safe step size is $1/1000$. This gives us the correct bound:

$$
\alpha \leq \frac{1}{\rho(\mathbf{H})} = \frac{1}{\max_i |\lambda_i|}
$$

And the full second-order Taylor approximation using the Hessian is:

$$
f(\mathbf{x} + \boldsymbol{\delta}) \approx f(\mathbf{x}) + \nabla f^T \boldsymbol{\delta} + \frac{1}{2}\boldsymbol{\delta}^T \mathbf{H} \boldsymbol{\delta}
$$

If we step along a unit eigenvector $\mathbf{v}_i$ of the Hessian (normalised so that $\|\mathbf{v}_i\| = 1$) with eigenvalue $\lambda_i$, then $\mathbf{v}_i^T \mathbf{H} \mathbf{v}_i = \lambda_i$. (This identity relies on the eigenvector being unit-length. If it were not normalised, we would get $\lambda_i \|\mathbf{v}_i\|^2$ instead.) So a large positive $\lambda_i$ means the function rises sharply along that direction, so the valley is narrow and steep-sided in that direction, and we should take small steps. A small positive $\lambda_i$ means the function is nearly flat, the valley is wide and gentle, and we can afford to stride. The eigenvalues literally tell us the shape of the bowl (or saddle) we are sitting in, direction by direction.

---

## 11  Putting It All Together

Let us trace the thread, the same way we did in the linear algebra post.

We started with **partial derivatives**, the simple idea of differentiating one variable at a time while freezing the rest. We extended the single-variable chain rule to the **multivariable chain rule**, which tells us how changes propagate through a chain of dependent functions by summing every path from input to output. This chain rule is the mathematical backbone of backpropagation.

We packaged all the partial derivatives into the **Jacobian**, a matrix that gives us the linear approximation $f(\mathbf{x} + \mathbf{h}) \approx f(\mathbf{x}) + \mathbf{J}\mathbf{h}$ for vector-valued functions. The Jacobian is a machine: you query it with a direction, and it tells you the resulting change. But for the special case of a scalar loss function, we needed more than a machine. We needed an *arrow*, a direction we could walk in. By transposing the Jacobian's row vector into the **gradient** column vector, we moved from the dual space into the original vector space, enabling inner products and the proof that the gradient points toward steepest ascent.

The **Taylor series** revealed that our linear approximation is just the first term in a hierarchy of increasingly accurate approximations. The second-order term introduced the **Hessian**, whose eigenvalues decompose the multi-directional curvature of the loss landscape into independent components, distinguishing bowls (all positive) from hills (all negative) from saddle points (mixed signs), generalising the familiar 2D second derivative test to arbitrary dimensions.

**Gradient descent** uses the gradient to step downhill: $\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \nabla f$. The squared norm guarantees descent, and the gradient direction guarantees it is the steepest. **Lipschitz continuity** of the gradient imposes a speed limit on how fast the slope can change, bounding the safe step size at $\alpha \leq 1/L$ [4]. And the Hessian's spectral radius (largest absolute eigenvalue) provides a concrete handle on $L$, closing the loop between curvature, step size, and convergence.

Together, these tools give us everything we need to train a neural network from scratch: compute the loss, compute the gradient via the chain rule (backpropagation [6]), step in the direction of steepest descent, and control the step size using curvature information. No magic, no black boxes, just calculus and linear algebra working together.

---

*In the next post, we will put all of this into practice. We will build a simple neural network from scratch, train it using gradient descent, and watch it learn to predict an "X" on a 3×3 screen. No frameworks, no libraries, just the math we covered in these two posts.*

---

## References

[1] M. P. Deisenroth, A. A. Faisal, and C. S. Ong, *Mathematics for Machine Learning*, Cambridge University Press, 2020. Chapters 5 (Vector Calculus) and 7 (Continuous Optimization) cover the Jacobian, gradient, Taylor series, and gradient descent presented in this post. Freely available at [mml-book.com](https://mml-book.com).

[2] Y. N. Dauphin, R. Pascanu, C. Gulcehre, K. Cho, S. Ganguli, and Y. Bengio, "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization," *Advances in Neural Information Processing Systems (NeurIPS)*, 2014. The argument that saddle points vastly outnumber local minima in high-dimensional loss landscapes (Section 10, "Why This Matters for Neural Network Training") is based on this paper.

[3] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," *International Conference on Learning Representations (ICLR)*, 2015. Referenced in Section 9 as an example of an adaptive optimiser that estimates local curvature to set per-parameter learning rates.

[4] Y. Nesterov, *Introductory Lectures on Convex Optimization: A Basic Course*, Springer, 2004. The descent lemma and the gradient descent convergence guarantee with step size $\alpha \leq 1/L$ (Section 9) originate from this work.

[5] S. Boyd and L. Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004. A standard reference for the Lipschitz gradient condition and its role in convergence analysis. Freely available at [stanford.edu/~boyd/cvxbook](https://stanford.edu/~boyd/cvxbook/).

[6] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, "Learning representations by back-propagating errors," *Nature*, vol. 323, pp. 533–536, 1986. The paper that popularised backpropagation, which is the practical application of the multivariable chain rule (Section 4) to compute gradients in neural networks.

[7] Schwarz's theorem (also known as Clairaut's theorem on equality of mixed partials), referenced in Section 10, is a classical result in analysis. A clear treatment can be found in: W. Rudin, *Principles of Mathematical Analysis*, 3rd ed., McGraw-Hill, 1976, Theorem 9.41.

[8] G. Hinton, "Neural Networks for Machine Learning, Lecture 6e: RMSProp," Coursera, 2012. RMSProp, referenced in Section 9 alongside Adam and Adagrad, was proposed in these unpublished lecture slides.

[9] J. Duchi, E. Hazan, and Y. Singer, "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization," *Journal of Machine Learning Research*, vol. 12, pp. 2121–2159, 2011. Adagrad, referenced in Section 9 as an adaptive optimiser.

[10] A. Cauchy, "Méthodes générales pour la résolution des systèmes d'équations simultanées," *Comptes Rendus de l'Académie des Sciences*, vol. 25, pp. 536–538, 1847. The original proposal of the method of steepest descent (gradient descent), referenced in Section 8.

[11] H. Robbins and S. Monro, "A Stochastic Approximation Method," *The Annals of Mathematical Statistics*, vol. 22, no. 3, pp. 400–407, 1951. The foundational paper on stochastic approximation, which underlies stochastic gradient descent (SGD), referenced in Section 10.
