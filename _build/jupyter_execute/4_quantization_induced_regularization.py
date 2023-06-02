#!/usr/bin/env python
# coding: utf-8

# # Quantization-Induced Regularization
# 
# During our [minifloat adventure](3_wonderful_minifloats.ipynb) it became clear that low-precision arithmetic produces quantization noise so strong that it must change statistical properties of ML models. This led me to a thought: in the simplest case of the linear regression how this quantization noise would affect the model? Here's a small note on that. *(Hint is in the title)*.

# # Formulation
# 
# In general, modern computers cannot represent *true* real numbers in all their Platonic nature, they are always quantized. Quantization makes a true value $x$ unobservable. A quantized value $q$ is not a real number, it is just a label for an infinite-sized equivalence class consisting of all numbers from some neighborhood $[q-a, q+b)$. When provided a $q$ the true number can be anything from this range and without any prior knowledge any number can result in $q$ class equally likely. We can model this uncertainty by a random variable $\varepsilon \sim U[-a, b]$ independent of $q$:
# 
# $$x = q + \varepsilon$$
# 
# Parameters $a$ and $b$ depend on the quantization scheme used. One may point out that quantization isn't random, and it surely isn't. But the point is not in the randomness of values but in the *distribution* of the quantization errors: without any prior knowledge taking a random number from the equivalent class represented by $q$ results in a uniform distribution.
# 
# A linear regression is formulated as:
# 
# $$ y = x^T w,\quad (x, y) \in D $$
# 
# Modifying it to account for quantization noise would produce the following equation:
# 
# $$ y + \varepsilon_y = \left( x + \varepsilon_x \right)^T w $$
# 
# Parameter vector $w$ is unmodified because it is unobserved and estimated based on data, which is quantized. So let's see what this means for integer and floating-point quantization.

# # Integer Quantization
# 
# Under integer quantization $\varepsilon \sim U\left[-\frac{d}{2}, \frac{d}{2}\right]$ where $d$ is a quantization step. We can notice that:
# 
# $$ \mathbb{E}[\varepsilon] = 0 $$
# $$ \sigma_\varepsilon^2 = \frac{1}{12} d^2 $$
# 
# Now recall a least-squared loss function for linear regression:
# 
# $$ L(w) = \mathbb{E}_{D} \left[ \left( y + \varepsilon_y - \left( x + \varepsilon_x \right)^T w \right)^2 \right] $$
# 
# Let's perform some algebra real quick:
# 
# $$ z^2 = \left( \left( y - x^T w \right) + \varepsilon_y - \varepsilon_x^T w \right)^2 = \\
#     = \left( y - x^T w \right)^2 + \varepsilon_y^2 + w^T \varepsilon_x \varepsilon_x^T w + 2 \left( y - x^T w \right) \varepsilon_y  - 2 \left( y - x^T w \right) \varepsilon_x^T w - 2 \varepsilon_y \varepsilon_x^T w $$
#     
# Now let's take and expectation with respect to $\varepsilon$ and use the fact $\varepsilon_x$ and $\varepsilon_y$ are independent of each other and anything else:
# 
# $$ \mathbb{E}_{\varepsilon} \left[ z^2 \right] 
#     = \left( y - x^T w \right)^2 
#         + \mathbb{E} \left[ \varepsilon_y^2 \right]
#         + w^T \mathbb{E} \left[ \varepsilon_x \varepsilon_x^T \right] w + \\
#         + 2 \left( y - x^T w \right) \mathbb{E} \left[ \varepsilon_y \right] 
#         - 2 \left( y - x^T w \right) \mathbb{E} \left[ \varepsilon_x \right]^T w 
#         - 2 \mathbb{E} \left[ \varepsilon_y \right] \mathbb{E} \left[ \varepsilon_x \right]^T w = \\
#     = \left( y - x^T w \right)^2 + \sigma^2 + \sigma^2 w^T w = \left( y - x^T w \right)^2 + \sigma^2 + \sigma^2 \| w \|^2 $$
#     
# Putting that back into the loss function produces:
# 
# $$ L(w) = \mathbb{E}_{D} \left[ z^2 \right] = \mathbb{E}_{D} \left[ \left( y - x^T w \right)^2 \right] + \sigma^2 \| w \|^2 + \sigma^2 $$
# 
# Adding a constant to a loss function doesn't change the minimum, so the $\sigma^2$ term can be dropped:
# 
# $$ L(w) = \mathbb{E}_{D} \left[ \left( y - x^T w \right)^2 \right] + \sigma^2 \| w \|^2 $$
# 
# Which is the same least-squares problem but with $L_2$ regularization with regularization parameter:
# 
# $$ \lambda = \frac{1}{12} d^2 $$
# 
# This is kinda expected as $L_2$ regularization is equivalent to adding zero-mean noise to the regressors, which we precisely did by our quantization scheme.

# # Floating-Point Quantization
# 
# Quantization noise distribution under FP quantization is a bit more complex. Given a floating point representation $x = M \cdot 2^E$ the distribution is $\varepsilon \sim U \left[ -\frac{a}{2}, \frac{a}{2} \right]$, where:
# 
# $$ a = p \cdot 2^{\lceil \log_2 x \rceil} $$
# 
# Here $p$ is the minimal representable normal FP number. This distribution has zero mean and variance $\sigma^2 = \frac{1}{12} p^2 4^{\lceil \log_2 x \rceil}$
# 
# This is because the absolute error of a floating point number doubles with each power of two. This exact formulation is unwieldy because of its nonlinear dependence on $x$, which will complicate further calculations quite a bit. Instead, we're gonna pretend that $ \lceil \log_2 x \rceil \approx \log_2 x $ and use the following approximation:
# 
# $$ a \approx p \cdot x $$
# 
# Now we can decompose this distribution as $\varepsilon = x \cdot \delta$, where $ \delta \sim U \left[ -\frac{p}{2}, \frac{p}{2} \right]$ and is now independent of $x$. This transforms our quantized linear regression formula into this:
# 
# $$ y + y \delta_y = \left( x + x \delta_x \right)^T w $$ 
# 
# The loss function now looks like this:
# 
# $$ L(w) = \mathbb{E}_{D} \left[ \left( y + y \delta_y - \left( x + x \delta_x \right)^T w \right)^2 \right] $$
# 
# We're gonna expand the brackets and take the expectation with respect to $\delta$ the same as before, so I leave it to you as an exercise. After doing all that we obtain the following:
# 
# $$ \mathbb{E}_{\varepsilon} \left[ z^2 \right] = \left( y - x^T w \right)^2 + \sigma^2 y^2 + \sigma^2 w^T x x^T w $$
# 
# Putting that back into the loss function gives:
# 
# $$ L(w) = \mathbb{E}_{D} \left[ z^2 \right] 
#     = \mathbb{E}_{D} \left[ \left( y - x^T w \right)^2 \right] + \sigma^2 \mathbb{E}_{D} \left[ y^2 \right] + \sigma^2 w^T \mathbb{E}_{D} \left[ x x^T \right] w = \\
#     = \mathbb{E}_{D} \left[ \left( y - x^T w \right)^2 \right] + \sigma^2 \mathbb{E}_{D} \left[ y^2 \right] + \sigma^2 w^T \Sigma_X w $$
#     
# Notice that we can always transform our problem by standardizing $y$ as well as whitening $x$, so $\mathbb{E}_{D} \left[ y^2 \right] = 1$ and $\Sigma_X = I$. And again, by noticing that addition of a constant doesn't change the minimum of a function, we get:
# 
# $$ L(w) = \mathbb{E}_{D} \left[ \left( y - x^T w \right)^2 \right] + \sigma^2 \| w \|^2 $$
# 
# So again we get an $L_2$-regularized model but under the assumption that the data is whitened and normalized. The regularization strength in this case bacomes:
# 
# $$ \lambda = \frac{1}{12} p^2 $$
# 
# I was looking for ways to verify these results, and the *proper* way to do this is to fit two linear models - the one on the "clean" data with $L_2$ prior, and another one on the noisy data without $L_2$ prior - and run a test for coefficient equality. But unfortunately statistical tests for equality of linear regression coefficients I managed to find aren't suited for this kind of a problem. Wald test compares regression coefficients to a hypothesized value, not another parameter. Chow test requires the same model to be fitted on multiple data groups but we have different ones. Z-test requires independent data samples for two models, but our data is definitely not independent. So I decided to just brute-force my way out of this by running extensive simulations to demonstrate that the difference between two parameter vectors converges to zero, which it does indeed:

# In[1]:


import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.decomposition import PCA
from tqdm import tqdm

np.random.seed(42)

DATA_ITER = 10
DATA_SIZE = 1000
DATA_DIM = 10

NOISE_ITER = 1000
NOISE_SIGMA = 1e-1
ALPHA = 1000 * NOISE_SIGMA**2

def generate_data(n, d):
    X = PCA(whiten = True).fit_transform(np.random.randn(n, d))
    w = np.random.randn(d)
    y = np.dot(X, w)
    y = (y - y.mean()) / y.std()
    return X, y

def average_model_over_noise(X, y, sigma, n):
    coef = []
    for _ in range(n):
        Z = X + sigma*X*np.random.randn(*X.shape)
        estimate = LinearRegression(fit_intercept = False).fit(Z, y).coef_
        coef.append(estimate)
    coef = np.stack(coef).mean(axis = 0)
    return coef

error_norm = []
for i in range(DATA_ITER):
    X, y = generate_data(DATA_SIZE, DATA_DIM)
    coef_reg = Ridge(alpha = ALPHA, fit_intercept = False).fit(X, y).coef_
    coef_noise = average_model_over_noise(X, y, NOISE_SIGMA, NOISE_ITER)
    error_norm.append(np.linalg.norm(coef_reg - coef_noise))
    
np.std(error_norm)


# # Implications
# 
# This implicit regularization doesn't manifest itself unless there are some extreme conditions. For example, even standard FP32 numbers produce this effect, but it is really tiny. The minimal normal FP32 number is ~10<sup>-38</sup>, which makes the regularization parameter about ~10<sup>-77</sup> give or take. But for half-precision numbers it would be ~10<sup>-10</sup> already. Where it becomes really prominent is at integer quantization over large ranges and low-precision FP quantization. 
# 
# Let's consider an 8-bit integer quantizer with range $R$. The quantization step is equal to $d = \frac{1}{256} R$ and regularization strength therefore being $\lambda = \frac{1}{3072} R^2$. So even a range of 1 will have regularization strength comparable to that used in practice. 
# 
# The same is true for low-precision floating-point numbers. An FP8 number (in a reasonable configuration) would induce regularization of order ~10<sup>-5</sup> which is on a lower side of values used in many papers. But FP4 numbers raise it to ~0.01 which is kinda significant. 
# 
# It should be noted that at such low precision the approximation for the quantization noise distribution we made earlier is very inaccurate so the exact numbers are almost certainly different. But as a sanity check I ran a quick test by fitting a model to "true" FP32 data and then to the same data but quantized down to FP4 precision. I then compared the coefficient vectors between the two and observed a consistent 1.5% decrease of the norm for the quantized version which is exactly what $L_2$ regularization does. 

# # Conclusion
# 
# Today we've explored the impact of an aggressive quantization on "learning" properties of a linear regression. We've established analytically that a linear model fitted to a quantized data is $L_2$-regularized in relation to the *"true"* data, meaning that the norm of the coefficient vector for a quantized model is smaller. We've also established a relation between quantization accuracy and regularization strength, which while being mostly negligible can become pretty significant at extremely low precisions. 
# 
# Unfortunately, it is unclear at the moment if there's a regularization effect of quantization in a deep learning setting, where all these FP8s, FP4s and INT8s make more sense, so I guess it's a topic for a future post. Have a good night.
