#!/usr/bin/env python
# coding: utf-8

# # Adam vs. Grad Clip
# 
# *Jun 30, 2023*
# 
# ----------------------------------
# 
# I was playing with [DreamBooth](https://github.com/TheLastBen/fast-stable-diffusion) one day. In case you missed it, it's a tool which fine-tunes Stable Diffusion text-to-image model on your photos, so you can then create beautiful portraits. And I believe it is truly a *"future's future future"* except one thing: training time. To create a custom model you need to wait for about 40 minutes on a Tesla T4 which is an exact opposite of fun.  
# 
# So I started thinking on how to speed up the fine-tuning, ideally without interfering with the training procedure too much and preserve image quality as much as possible. I ran a training code through a `line_profiler` to find out that it is pretty well optimized from the start. But not perfectly. Besides some other things a particular [line](https://github.com/TheLastBen/diffusers/blob/7b633fd9c441fc1b0e6cb75524cd568243523b1c/examples/dreambooth/train_dreambooth.py#L723) (it's still there) caught my eye:
# ```
# Line   №  Hits        Time   Per Hit  % Time  Line Contents
# ==============================================================
#      723   100  41914431.8  419144.3    10.1  accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
# ```
# A substantial chunk of training time (10%) was spent on clipping gradient norms, and I had a great desire to remove it. So my first idea was to check if the gradient ever gets clipped, and if it doesn't then this line can be safely removed. However I have found exactly the opposite: the gradient was always getting clipped, because its raw norm was ~10<sup>3</sup> while the threshold was set to 1. But does it still make any difference?

# # Adam Refresher
# 
# There's plenty of information about Adam optimizer on the [Internet](https://letmegooglethat.com/?q=how+Adam+optimizer+works) including the original [paper](https://arxiv.org/abs/1412.6980). So here's a quick recap. 
# 
# A formula for the weight update in the Adam algorithm is:
# 
# $$ m_t = \frac{1}{1 - \beta_1^t} \Bigl( \beta_1 m_{t-1} + \left( 1 - \beta_1 \right) g_t \Bigr) $$
# $$ v_t = \frac{1}{1 - \beta_2^t} \Bigl( \beta_2 v_{t-1} + \left( 1 - \beta_2 \right) g_t^2 \Bigr) $$
# $$ \Delta w_t = -\lambda \frac{m_t}{\sqrt{v_t + \varepsilon}} $$
# 
# We are interested in the $v_t$ part of the formula. Here $v_t$ is sort of an element-wise variance of the gradient $g_t$. Dividing by $v_t$ we force weight updates $\Delta w_t$ to have a fixed variance with some EMA fluctuations. 

# # So What?
# 
# Gradients in neural networks have almost zero mean and so are weight updates. And when each component of the weight update has fixed variance, then its norm is also constant on average, no matter what the variance of the gradient is. So no matter what is the variance of the gradient, weight updates become normalized anyway. Which means that gradient normalization is redundant when used alongside Adam. Of course, it has *some* impact - skews first few steps until EMA statistics accumulate. But is it really that important? Depends on the problem you're trying to solve. In most cases you usually don't care about Adam warm-up at all. In the case of the DreamBooth, fine-tuning takes not many steps, and this scaling can make a difference on the final result. But you can achieve a similar effect by changing the learning rate. 
# 
# Testing this hypothesis is very simple: just comment the corresponding line of code and compare the results. And if you do that as I did, you'll see that the quality of generated images doesn't change while you shave off 10% of execution time. 

# # Conclusion
# 
# Gradient norm clipping is a kinda costly operation, so don't add it to your model unless you really have to (numerical issues and whatnot). Adam will take care of variance scaling anyway. 
# 
# Have a good night.
