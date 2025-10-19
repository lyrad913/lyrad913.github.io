---
title: Diffusion Model Application
publish: true
creation_date: 2025-08-13 08:00
modification_date: 2025-08-13 08:00
categories:
tags:
  - generative_model
  - diffusion
  - guidance
date: 2025-08-12T23:00:48.382Z
lastmod: 2025-10-19T13:47:39.392Z
---
{{< katex />}}

$p(x)$, 즉, 데이터의 분포를 모델링하는 것보단 $p(x|y)$를 모델링하는 것이 유용한 경우가 많다.  $y$ 는 텍스트라든지, 이미지라든지, 레이블이 될 수 있다.

## Conditional Diffusion Model

<div>
$$
p_{\theta}(x_{t-1}|x_{t},y) = N(x_{t-1}; \mu_{\theta}(x_{t}, t, y), \sigma_{q}^2I)
$$
</div>

## Improvement(Score function)

<div>
$$
\epsilon \approx -\sqrt{ 1-\bar{\alpha}_{t} }\nabla_{x_{t}}\log p(x_{t})
$$
</div>

여기서 $\nabla_{x_{t}}\log p(x_{t})$를 score 또는 score function이라 부른다. 노이즈와 점수가 상수배 차이가 나므로, 점수 자체를 학습하는 신경망도 가능하다.

### 유도

<div>
$$
\begin{align}
q(x_{t}|x_{0})  & = N(x_{t}; \sqrt{ \bar{\alpha}_{t}  }x_{0}, (1-\bar{\alpha}_{t})I) \\
 & \text{(by Tweedie's Formula)} \\
\mathbb{E}[\sqrt{ \bar{\alpha}_{t} }x_{0}|x_{t}]  & = x_{t} + (1-\bar{\alpha}_{t})\nabla_{x_{t}}\log p(x_{t}) \\
 & \Leftrightarrow \mathbb{E}[x_{t}-\sqrt{ 1-\bar{\alpha}_{t} }\epsilon|x_{t}]  = x_{t} + (1-\bar{\alpha}_{t})\nabla_{x_{t}}\log p(x_{t}) \\
 & \Leftrightarrow\mathbb{E}[x_{t}|x_{t}] - \mathbb{E}[\sqrt{ 1-\bar{\alpha}_{t} }\epsilon|x_{t}] = x_{t} + (1-\bar{\alpha}_{t})\nabla_{x_{t}}\log p(x_{t}) \\
 & \Leftrightarrow x_{t}- \mathbb{E}[\sqrt{ 1-\bar{\alpha}_{t} }\epsilon|x_{t}] = x_{t} + (1-\bar{\alpha}_{t})\nabla_{x_{t}}\log p(x_{t}) \\
 & \Leftrightarrow -\sqrt{ 1-\bar{\alpha}_{t} }\mathbb{E}[\epsilon|x_{t}] = (1-\bar{\alpha}_{t})\nabla_{x_{t}}\log p(x_{t}) \\
 & \text{(by Monte Carlo)} \\
 & 
\epsilon \approx -\sqrt{ 1-\bar{\alpha}_{t} }\nabla_{x_{t}}\log p(x_{t})
\end{align}
$$
</div>

> \[!Twedie's Formula]-
>
> <div>
> $$
> x \sim N(x; \mu, \Sigma)\text{ then }\mathbb{E}[\mu|x] = x + \Sigma \nabla_{x}\log p(x)
> $$
> </div>

### Classifier Guidance

![attachments/Pasted image 20250813081721.png](/images/cerebro/Artificial%20Intelligence/Neural%20Network/attachments/Pasted%20image%2020250813081721.png)

<div>
$$
\begin{align}
p(x_{t}|y)  & = \frac{p(x_{t})p(y|x_{t})}{p(y)} \\
 & \text{양변 로그, 미분} \\
\nabla_{x_{t}}\log p(x_{t}|y)  & = \nabla_{x_{t}}\log p(x_{t}) + \nabla_{x_{t}} \log p(y|x_{t}) + \nabla_{x_{t}}\log p(y) \\
\text{(조건부 점수)}  & = \text{(점수; diffusion)}  \\
& +  \text{(분류기 로그 가능도의 기울기; 분류기 신경망으로 계산 가능)} \\
 &  + 0 \\ 
\end{align}
$$
</div>

<div>
$$
	\nabla_{x_{t}}\log(x_{t}|y) \approx s_{\theta}(x_{t}, t) + \gamma \nabla_{x_{t}}\log p_{\phi}(y|x_{t}) 
$$
</div>

### Classifier-Free Guidance

![attachments/Pasted image 20250813083227.png](/images/cerebro/Artificial%20Intelligence/Neural%20Network/attachments/Pasted%20image%2020250813083227.png)

<div>
$$
\begin{align}
\nabla_{x_{t}}\log p(x_{t}|y)  & = \nabla_{x_{t}}\log p(x_{t}) + \gamma\nabla_{x_{t}} \log p(y|x_{t}) \\
 & = \nabla_{x_{t}}\log p(x_{t}) + \gamma\{\nabla_{x_{t}}\log p(x_{t}|y) + \nabla_{x_{t}}\log p(y) - \nabla_{x_{t}}\log p(x_{t})\} \\
 & = \nabla_{x_{t}}\log p(x_{t}) + \gamma\{\nabla_{x_{t}}\log p(x_{t}|y) - \nabla_{x_{t}}\log p(x_{t})\}
\end{align}
$$
</div>

조건 없는 점수, 조건부 점수 둘 다 하나의 모델로 해결할 수 있다.

<div>
$$
\nabla_{x_{t}}\log p(x_{t}|y) \approx s_{\theta}(x_{t}, t, \emptyset) + \gamma\{s_{\theta}(x_{t}, t, y) - s_{\theta}(x_{t}, t, \emptyset)\}
$$
</div>

### Stable Diffusion

![attachments/Pasted image 20250813083648.png](/images/cerebro/Artificial%20Intelligence/Neural%20Network/attachments/Pasted%20image%2020250813083648.png)

![attachments/Pasted image 20250813083655.png](/images/cerebro/Artificial%20Intelligence/Neural%20Network/attachments/Pasted%20image%2020250813083655.png)

![attachments/Pasted image 20250813083705.png](/images/cerebro/Artificial%20Intelligence/Neural%20Network/attachments/Pasted%20image%2020250813083705.png)
