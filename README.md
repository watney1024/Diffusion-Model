# Diffusion-Model

## 2025年1月11日
在B站找了一些视频:
[扩散模型 Diffusion Model 1-1 概述](https://www.bilibili.com/video/BV1fU4y1i7kK/?share_source=copy_web&vd_source=5dc089148483e98d94828c5bf4ea5929)  
[2024.11.14组会-去噪扩散模型（DDPM)]( https://www.bilibili.com/video/BV112U5YmEQL/?share_source=copy_web&vd_source=5dc089148483e98d94828c5bf4ea5929)  
[2024.11.28组会-改进的去噪扩散模型（DDIM )]( https://www.bilibili.com/video/BV1gGzhYCENp/?share_source=copy_web&vd_source=5dc089148483e98d94828c5bf4ea5929)  
[【论文精读】Diffusion Model 开山之作DDPM](https://www.bilibili.com/video/BV1WD4y157u3/?share_source=copy_web&vd_source=5dc089148483e98d94828c5bf4ea5929)  
[【较真系列】讲人话-Diffusion Model全解(原理+代码+公式)](https://www.bilibili.com/video/BV19H4y1G73r/?share_source=copy_web&vd_source=5dc089148483e98d94828c5bf4ea5929)   
下载了DDPM和DDIM论文，第一个视频看到了1-4，大致知道了DDPM的过程

## 2025年1月18日
跑了下DDPM的代码，数据集是cifar10,250个epoch在服务器上跑了4个小时。

## 2025年1月19日
写了一部分DDPM的公式分析，完成了扩散和重建阶段的数下公式推导，极大似然部分完成了一部分。由于公式比较多，我是用latex写的，所以只上传了pdf文件。





diffusion入门攻略（更新中）
和diffusion关系不大，了解即可
https://www.youtube.com/watch?v=Tk5B4seA-AU&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=25 
Unsupervised Learning - Auto-encoder
https://www.youtube.com/watch?v=YNUek8ioAJk&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=26 
Unsupervised Learning - Deep Generative Model (Part I) —— pixelRNN
https://www.youtube.com/watch?v=8zomhgKrsmQ&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=27 
Deep Generative Model (Part II) —— VAE, GAN

图像生成模型简介（科普性质）：
https://www.youtube.com/watch?v=z83Edfvgd9g 
https://www.youtube.com/watch?v=azBugJzmz-o 
https://www.youtube.com/watch?v=JbfcAaBT66U 

diffusion原理（涉及数学推导，可先不求甚解）
https://www.youtube.com/watch?v=ifCDXFdeaaM 
https://www.youtube.com/watch?v=73qwu77ZsTM 
https://www.youtube.com/watch?v=m6QchXTx6wA 
https://www.youtube.com/watch?v=67_M2qP5ssY 

Deep Unsupervised Learning using Nonequilibrium Thermodynamics
2024/04/15-2024/04/21 
https://www.bilibili.com/video/BV19v4y1C7De/ 
视频中有不少推导方面的错误，但是可以看一下大致思路帮助快速理解
Denoising Diffusion Probabilistic Models
https://spaces.ac.cn/archives/9119 
https://spaces.ac.cn/archives/9152 
https://spaces.ac.cn/archives/9164 (最容易理解)
Understanding Diffusion Models: A Unified Perspective
补充材料，从多个角度推导出了diffusion及其等价性（P15-P17）
DENOISING DIFFUSION IMPLICIT MODELS
https://spaces.ac.cn/archives/9181 
SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS


## 2025年3月28日
上传了训练的文件，包括了ddpm和ddim，目前是在mnist上训练。