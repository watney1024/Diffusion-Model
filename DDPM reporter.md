# 训练过程

![Desktop Screenshot 2025.03.18 - 14.48.16.31](D:/code/Pycharm/Diffusion-Model/Desktop%20Screenshot%202025.03.18%20-%2014.48.16.31.png)

## 正向过程

通过不断给图片加上高斯噪声，最后图片会变成一张高斯噪声。              

$$ q(x_{1:T} \vert x_0) = \prod_{t=1}^Tq(x_t \vert x_{t-1}),\ \ \ \text{其中}\ \ \ q(x_t \vert x_{t-1})=\mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t \mathbf{I})$$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

$ x_t = \sqrt{1-\beta_t} \cdot x_{t-1}+\sqrt{\beta_t} \cdot \epsilon, \text{其中}\beta_t \in (0,1)$

令$\alpha_t = 1 - \beta_t$ ，令$\bar {\alpha_t} = \alpha_1 \cdot \alpha_2 \ldots \alpha_t$并且经过迭代，可以得到新的递推式

$x_t = \sqrt{\bar{\alpha_t}} \cdot x_0 + \sqrt{1-\bar{\alpha_t}} \cdot \epsilon_0$

当t足够大时，可以认为$x_t \sim \mathcal{N}(0,1)$



## 反向过程

通过unet来预测噪声，逐步给图片去噪。

$$p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^{T}p_{\theta}(x_{t-1} \vert x_t),\ \ \ \text{其中}\ \ \ p_{\theta}(x_{t-1} \vert x_t) = \mathcal{N}(x_{t-1} ; \mu_{\theta}(x_t, t), \Sigma_\theta(x_t, t))$$

$x_{t-1} \sim q(x_{t-1} | x_t)  \propto q(x_{t-1}) \cdot q(x_t | x_{t-1})$



## 损失函数

$$ \mathbb{E}_q[-\log p_\theta(x_0)] \leq \mathbb{E}_q[-\log {{p_\theta(x_{0:T})}\over{q(x_{1:T} \vert x_0)}}]=\mathbb{E}_q[-\log p(x_T)-\sum_{t\geq1} \log {{ {p_\theta(x_{t-1} \vert x_t)} }\over{q(x_t \vert x_{t-1})}} ]=L $$

$$\begin{align}L &= \mathbb{E}_q [ D_{KL}(q(x_T|x_0) || p(x_T)) + \sum_{t>1}D_{KL}(q(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t))-\log p_\theta (x_0|x_1) ] \end{align}$$

其中$q(x_{t-1} \vert x_t, x_0)=\mathcal{N}(x_{t-1};\tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t)$ ， $\tilde{\mu}_t(x_t, x_0)=\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1}) }{1-\bar{\alpha}_t}x_t$，$\tilde{\beta}_t=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$

## 训练过程

+ 从数据分布中采样初始数据 $\mathbf{x}_0 \sim q(\mathbf{x}_0)$

+ 随机选择时间 t
+ 生成噪声$\epsilon \sim \mathcal{N}(0,1)$
+  对于$\mathcal{L} = \frac{1}{2 \sigma_q^2(t)} \frac{(1-\alpha_t)^2}{\left(1-\bar{\alpha}_t\right)\alpha_t}\left[\left\|f_\boldsymbol{\theta}(\boldsymbol{x}_t, t)-\boldsymbol{\varepsilon}_t\right\|_2^2\right]$使用梯度下降法
+ 重复直到收敛



## 采样过程

+ 对于$x_T \sim \mathcal{N}(0,1)$

+ 对于 $t = T \ldots 1$循环
+ $z \sim \mathcal{N}(0,1)$
+ $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{1- \alpha_t}{\sqrt{1-\bar\alpha_t}} \cdot \epsilon_\theta(x_t, t))+ \sigma_t z$     这里的$\epsilon_\theta$是预测出的噪声，$z$是随机噪声，为了增加随机性。



## 具体代码

```python
class DiffusionProcess():
    def __init__(self, beta_1, beta_T, T, diffusion_fn, device, shape):
        '''
        beta_1        : beta_1 of diffusion process
        beta_T        : beta_T of diffusion process
        T             : step of diffusion process
        diffusion_fn  : trained diffusion network
        shape         : data shape
        '''

        self.betas = torch.linspace(start = beta_1, end=beta_T, steps=T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start = beta_1, end=beta_T, steps=T), dim = 0).to(device = device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])
        self.shape = shape
        
        self.diffusion_fn = diffusion_fn
        self.device = device

    
    def _one_diffusion_step(self, x):
        '''
        x   : perturbated data
        '''
        for idx in reversed(range(len(self.alpha_bars))):
            noise = torch.zeros_like(x) if idx == 0 else torch.randn_like(x)
            sqrt_tilde_beta = torch.sqrt((1 - self.alpha_prev_bars[idx]) / (1 - self.alpha_bars[idx]) * self.betas[idx])
            predict_epsilon = self.diffusion_fn(x, idx)
            mu_theta_xt = torch.sqrt(1 / self.alphas[idx]) * (x - self.betas[idx] / torch.sqrt(1 - self.alpha_bars[idx]) * predict_epsilon)
            x = mu_theta_xt + sqrt_tilde_beta * noise
            yield x
    
    @torch.no_grad()
    def sampling(self, sampling_number, only_final=False):
        '''
        sampling_number : a number of generation
        only_final      : If True, return is an only output of final schedule step 
        '''
        sample = torch.randn([sampling_number,*self.shape]).to(device = self.device).squeeze()
        sampling_list = []
        
        final = None
        for idx, sample in enumerate(self._one_diffusion_step(sample)):
            final = sample
            if not only_final:
                sampling_list.append(final)

        return final if only_final else torch.stack(sampling_list)
```



## 问题

1. 原论文是预测噪声，因为预测原图的效果不好，需要自己做实验看看预测原图的效果

2. 做实验看一步采样和直接从$x_t$到$x_0$的效果
3. 