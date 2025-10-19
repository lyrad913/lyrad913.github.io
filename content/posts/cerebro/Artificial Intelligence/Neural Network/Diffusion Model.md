---
title: Diffusion Model
publish: true
creation_date: 2025-08-13 00:47
modification_date: 2025-08-13 00:47
categories:
tags:
  - generative_model
  - diffusion
date: 2025-08-12T15:47:52.317Z
lastmod: 2025-10-19T13:31:15.177Z
---
{{< katex />}}

[VAE](VAE)와 비교했을 때 다음의 차이가 있음

1. 관측 변수와 잠재 변수의 차원을 일치
2. 고정된 정규 분포를 따르는 노이즈를 인코더에 추가.

## 알고리즘

```
x_T ~ N(0, 1)
for t in [T, ..., 1]:
	epsilon ~ N(0, 1)
	if t = 1 then epsilon = 0
	x_{t-1} = mu_{theta}(x_t, t) + sigma_q(t)*epsilon
return x_0
```

## Diffusion Process

<div>
$$
q(x_{t}|x_{t-1}) = N(x_{t}; \sqrt{ 1-\beta_{t} }x_{t-1}, \beta_{t}I)
$$
</div>

### Noise Sampling

$\beta_{t}$  값은 설정된 값(선형, 지수, 코사인파 등등)

### 재매개변수화 트릭

<div>
$$
\begin{align}
\epsilon &\sim  N(0, 1) \\ 
x_{t} &= \sqrt{ 1-\beta_{t} }x_{t-1} + \sqrt{ \beta_{t} }\epsilon \\
\end{align}
$$
</div>

## Reverse Diffusion Process

![attachments/Pasted image 20250813005714.png](/images/cerebro/Artificial%20Intelligence/Neural%20Network/attachments/Pasted%20image%2020250813005714.png)

$p_{\theta}(x_{t-1}|x_{t})$를 모델링한다. 이를 위해 아래와 같이 적용함

<div>
$$
\begin{align}
& \hat{x}_{t-1} = NeuralNetwork(x_{t}, t; \theta) \\
& p_{\theta}(x_{t-1}|x_{t}) = N(x_{t-1}; \hat{x}_{t-1}, I)
\end{align}
$$
</div>

## Train Diffusion Model

<div>
$$
\text{Loss}(x_{0}; \theta) = \cfrac{1}{\sigma_{q}^2(t)}||\mu_{\theta}(x_{t}, t) - \mu_{q}(x_{t}, t)||^2
$$
</div>

* [Train Diffusion Model Derivation| Dervation](Train%20Diffusion%20Model%20Derivation%7C%20Dervation)

## Variation

### 바로 원본으로

<div>
$$
\mu_{q}(x_{t},x_{0}) = \cfrac{\sqrt{ \alpha_{t} }(1-\bar{\alpha}_{t-1})x_{t} + \sqrt{ \bar{\alpha}_{t-1} }(1-\alpha_{t})x_{0}}{1-\bar{\alpha}_{t}}
$$
</div>

따라서 $\mu_{\theta}$에 파라미터를 두는 대신 $x_{0}$자리에 파라미터를 두면 원본을 바로 복원할 수 있다.

<div>
$$
\mu_{\theta}(x_{t},x_{0}) = \cfrac{\sqrt{ \alpha_{t} }(1-\bar{\alpha}_{t-1})x_{t} + \sqrt{ \bar{\alpha}_{t-1} }(1-\alpha_{t})\hat{x}_{\theta}(x_{t}, t)}{1-\bar{\alpha}_{t}}
$$
</div>

<div>
$$
D_{KL}(q(x_{t-1}|x_{t}, x_{0})||p_{\theta}(x_{t-1}|x_{t})) = \frac{1}{2} \frac{1}{\sigma^2_{q}(t)} \left( \frac{\sqrt{ \bar{\alpha}_{t-1} }(1-\alpha_{t})}{1-\bar{\alpha}_{t}} \right)||x_{0}-\hat{x}_{\theta}(x_{t}, t)||^2
$$
</div>

### **노이즈 예측 신경망**

앞서 본 재매개변수화 트릭의 수식으로부터...

<div>
$$
x_{0} = \frac{x_{t}-\sqrt{ 1-\bar{\alpha}_{t} }\epsilon}{\sqrt{ \bar{\alpha}_{t} }}
$$
</div>

이를 $\mu_{q}(x_{t},x_{0})$에 대입하면

<div>
$$
\mu_{q}(x_{t},x_{0}) = \frac{1}{\sqrt{ \alpha_{t} }}\left( x_{t} - \frac{1-\alpha_{t}}{\sqrt{ 1-\bar{\alpha}_{t} }}\epsilon \right)
$$
</div>

원본 예측 모델과 동일하게 $\mu_{\theta}(x_{t}, t)$를 조정하면

<div>
$$
\mu_{\theta}(x_{t},x_{0}) = \frac{1}{\sqrt{ \alpha_{t} }}\left( x_{t} - \frac{1-\alpha_{t}}{\sqrt{ 1-\bar{\alpha}_{t} }}\epsilon_{\theta}(x_{t}, t) \right)
$$
</div>

<div>
$$
D_{KL}(q(x_{t-1}|x_{t}, x_{0})||p_{\theta}(x_{t-1}|x_{t})) = \frac{1}{2} \frac{1}{\sigma^2_{q}(t)} \left( \frac{(1-\alpha_{t})^2}{(1-\bar{\alpha}_{t})\alpha_{t}} \right)||\epsilon-\epsilon_{\theta}(x_{t}, t)||^2
$$
</div>

나머지는 상수니까 norm만 로스로 사용하면 된다.

### Application

[Diffusion-Model-Application](/posts/cerebro/Artificial-Intelligence/Neural-Network/Diffusion-Model-Application/)

[Diffusion-Model-Application]({{< relref "./Diffusion-Model-Application" >}})
