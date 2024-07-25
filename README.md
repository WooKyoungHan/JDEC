# JDEC: JPEG Decoding via Enhanced Continuous Cosine Coefficient
This repository contains the official implementation for JDEC introduced in the following paper:

[JDEC: JPEG Decoding via Enhanced Continuous Cosine Coefficient (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Han_JDEC_JPEG_Decoding_via_Enhanced_Continuous_Cosine_Coefficients_CVPR_2024_paper.pdf)


[arXiv](https://arxiv.org/abs/2404.05558)


[Project Page](https://wookyounghan.github.io/JDEC/)


## Overall Structure

![Overall Structure of Our JDEC](./static/images/Fig_4_ver_final_main.jpg)

JDEC consists of an encoder with group spectra embedding, a decoder, and a continuous cosine formulation. Inputs of JDEC are as follows: compressed spectra and quantization map. Note that our JDEC does not take images as input. JDEC formulates latent features into a trainable continuous cosine coefficient as a function of the block grid and forwards to INR. Therefore, each block shares the estimated continuous cosine spectrum.


TBD
