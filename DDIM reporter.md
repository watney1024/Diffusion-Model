## 算法思路

DDPM的一个缺点是推理速度慢，因为反向过程是根据马尔科夫链推导来的，因此只能一步一步迭代，如果要生成高质量的图片，T就会很大，此时进行推理的时候需要的时间也很长。DDIM不需要马尔科夫链，并且在训练时还是用DDPM，推理时采用DDIM即可。



## 反向过程

使用待定系数法，假设$P(x_{t-1}|x_t,x_0) \sim N(kx_0+mx_t, \sigma^2)$

$x_{t-1} = kx_0+mx_t+\sigma\epsilon$

由于$x_t = \sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$，代入上式可得

$x_{t-1} = kx_0+m[\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon]+\sigma\epsilon$

代入并合并同类项  

$x_{t-1} = k x_0 + m [\sqrt{\alpha_t} x + \sqrt{1 - \alpha_t} \epsilon] + \sigma \epsilon $
$x_{t-1} = (k + m \sqrt{\alpha_t}) x_0 + \epsilon' \epsilon' \sim N(0, m^2 (1 - \alpha_t) + \sigma^2)$



比较系数可得  

$k + m \sqrt{\alpha_t} = \sqrt{\alpha_{t-1}}$
$m^2 (1 - \alpha_t) + \sigma^2 = 1 - \alpha_{t-1}$

最后解出

$m = \frac{\sqrt{1 - \alpha_{t-1} - \sigma^2}}{\sqrt{1 - \alpha_t}} $
$k = \sqrt{\alpha_{t-1}} - \frac{\sqrt{1 - \alpha_{t-1} - \sigma^2}}{\sqrt{1 - \alpha_t}} \sqrt{\alpha_t}$

最后可以得出迭代公式

$$\large{x_{t-1}=\sqrt{\alpha_{t-1}} \underbrace{(\frac{x_t - \sqrt{1-\alpha_t}\epsilon_{\theta}^{(t)}(x_t)}{\sqrt{\alpha_t}})}_{\text{predicted }{x_0}} + \underbrace{\sqrt{1-\alpha_{t-1}-\sigma_t^2}\epsilon_{\theta}^{(t)}(x_t)}_{\text{direction pointing }{x_t}}+\underbrace{\sigma_t\epsilon_t}_{\text{random noise}}}$$













```python
class DiffusionProcessDDIM():
    def __init__(self, beta_1, beta_T, T, diffusion_fn, device, shape, eta, tau = 1, scheduling = 'uniform'):
        '''
        beta_1        : beta_1 of diffusion process
        beta_T        : beta_T of diffusion process
        T             : step of diffusion process
        diffusion_fn  : trained diffusion network
        shape         : data shape
        eta           : coefficient of sigma
        tau           : accelerating of diffusion process
        scheduling    : scheduling mode of diffusion process
        '''
        self.betas = torch.linspace(start = beta_1, end=beta_T, steps=T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start = beta_1, end=beta_T, steps=T), dim = 0).to(device = device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])

        self.shape = shape
        self.sigmas = torch.sqrt((1 - self.alpha_prev_bars) / (1 - self.alpha_bars)) * torch.sqrt(1 - (self.alpha_bars / self.alpha_prev_bars))
        self.diffusion_fn = diffusion_fn
        self.device = device
        self.eta = eta
        self.tau = tau
        self.scheduling = scheduling
        
    def _get_process_scheduling(self, reverse = True):
        if self.scheduling == 'uniform':
            diffusion_process = list(range(0, len(self.alpha_bars), self.tau)) + [len(self.alpha_bars)-1]
        elif self.scheduling == 'exp':
            diffusion_process = (np.linspace(0, np.sqrt(len(self.alpha_bars)* 0.8), self.tau)** 2)
            diffusion_process = [int(s) for s in list(diffusion_process)] + [len(self.alpha_bars)-1]
        else:
            assert 'Not Implementation'
            
        
        diffusion_process = zip(reversed(diffusion_process[:-1]), reversed(diffusion_process[1:])) if reverse else zip(diffusion_process[1:], diffusion_process[:-1])
        return diffusion_process
            

    
    def _one_reverse_diffusion_step(self, x):
        '''
        x   : perturbated data
        '''
        diffusion_process = self._get_process_scheduling(reverse = True)

        for prev_idx, idx in diffusion_process:
            self.diffusion_fn.eval()
            noise = torch.zeros_like(x) if idx == 0 else torch.randn_like(x)
            predict_epsilon = self.diffusion_fn(x, idx)
            sigma = self.sigmas[idx] * self.eta
            
            predicted_x0 = torch.sqrt(self.alpha_bars[prev_idx]) * (x - torch.sqrt(1 - self.alpha_bars[idx]) * predict_epsilon) / torch.sqrt(self.alpha_bars[idx])
            direction_pointing_to_xt = torch.sqrt(1 - self.alpha_bars[prev_idx] - sigma**2 ) * predict_epsilon
            x = predicted_x0 + direction_pointing_to_xt + sigma * noise

            yield x

    @torch.no_grad()
    def sampling(self, sampling_number, sample=None, only_final=False):
        '''
        sampling_number : a number of generation
        only_final      : If True, return is an only output of final schedule step 
        '''
        if sample==None:
            sample = torch.randn([sampling_number,*self.shape]).to(device = self.device)
            
        sampling_list = []
        
        final = None
        for sample in self._one_reverse_diffusion_step(sample):
            final = sample
            if not only_final:
                sampling_list.append(final)

        return final if only_final else torch.stack(sampling_list)

    
    @torch.no_grad()
    def probabilityflow(self, x, reverse):
        '''
        reverse : if True, backward(noise -> data) else forward(data -> noise)
        '''
        def reparameterize_sigma(idx):
            return torch.sqrt( (1-self.alpha_bars[idx]) / self.alpha_bars[idx] )
        def reparameterize_x(x, idx):
            return x / torch.sqrt(self.alpha_bars[idx])
        
        diffusion_process = self._get_process_scheduling(reverse = reverse)
        for idx_delta_t, idx in diffusion_process:
            self.diffusion_fn.eval()
            x_bar_delta_t = reparameterize_x(x, idx) + 0.5 * (reparameterize_sigma(idx_delta_t)**2 - reparameterize_sigma(idx)**2) / reparameterize_sigma(idx) * self.diffusion_fn(x, idx)
            x = x_bar_delta_t * torch.sqrt(self.alpha_bars[idx_delta_t])

        return x
```

