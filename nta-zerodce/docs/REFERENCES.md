# References

Primary author/publisher sources; accessed 2026-10-13. Dataset availability and
terms should be checked again at acquisition. Extensions in this repo are proposed
combinations, not claims from these papers. No reference implementation was copied.

## R1

Guo, C., Li, C., Guo, J., Loy, C. C., Hou, J., Kwong, S., and Cong, R.
Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement. CVPR 2020.
[Paper](https://arxiv.org/abs/2001.06826),
[CVF publication](https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.html).
Baseline curves, architecture and zero-reference objectives; not our extensions.

## R2

Li-Chongyi/Zero-DCE reference repository.
[README](https://github.com/Li-Chongyi/Zero-DCE),
[project page](https://li-chongyi.github.io/Proj_Zero-DCE.html).
Source of licence attribution and training-data discovery links. Reference-profile
loss reductions must be independently implemented and checked; do not import code
or reference weights. Code licence: CC BY-NC 4.0, academic/noncommercial.

## R3

Li, C., Guo, C., and Loy, C. C. Learning to Enhance Low-Light Image via
Zero-Reference Deep Curve Estimation. IEEE TPAMI (Zero-DCE++).
[Author project and publication links](https://li-chongyi.github.io/Proj_Zero-DCE++.html).
Efficiency motivation only; our light variant is independently specified.

## R4

Cai, J., Gu, S., and Zhang, L. Learning a Deep Single Image Contrast Enhancer
from Multi-Exposure Images. IEEE TIP 27(4), 2018, pp. 2049..2062.
[SICE author repository and dataset links](https://github.com/csjcai/SICE).
SICE multi-exposure data; use part 1 for our unpaired training protocol.

## R5

Wei, C., Wang, W., Yang, W., and Liu, J. Deep Retinex Decomposition for
Low-Light Enhancement. BMVC 2018.
[Author project / original LOL downloads](https://daooshee.github.io/BMVC2018website/).
Original LOL paired evaluation and Retinex-Net context.

## R6

Kroeger, T., Timofte, R., Dai, D., and Van Gool, L. Fast Optical Flow using
Dense Inverse Search. ECCV 2016.
[Paper](https://arxiv.org/abs/1603.03590),
[OpenCV DIS API](https://docs.opencv.org/4.x/d4/d43/classcv_1_1DISOpticalFlow.html).
DIS is our real-pair flow choice; darkness reliability is not guaranteed.

## R7

Farneback, G. Two-Frame Motion Estimation Based on Polynomial Expansion.
SCIA 2003, pp. 363..370.
[Publisher DOI](https://doi.org/10.1007/3-540-45103-X_50).
Alternative background flow method, not an added implemented backend in M0.

## R8

Immerkaer, J. Fast Noise Variance Estimation. Computer Vision and Image
Understanding 64(2), 1996, pp. 300..302.
[Publisher DOI](https://doi.org/10.1006/cviu.1996.0060).
Gaussian flat-region noise proxy; fine texture can be mistaken for noise.

## R9

Lai, W.-S., Huang, J.-B., Wang, O., Shechtman, E., Yumer, E., and Yang, M.-H.
Learning Blind Video Temporal Consistency. ECCV 2018, pp. 170..185.
[CVF paper page](https://openaccess.thecvf.com/content_ECCV_2018/html/Wei-Sheng_Lai_Real-Time_Blind_Video_ECCV_2018_paper.html),
[arXiv](https://arxiv.org/abs/1808.00449).
Related recurrent temporal-consistency work; unlike it, our proposed inference
explicitly computes optical flow. Do not transfer its speed claims to this project.

## R10

Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P. Image Quality
Assessment: From Error Visibility to Structural Similarity. IEEE TIP 2004.
[Author SSIM page](https://www.cns.nyu.edu/~lcv/ssim/).
SSIM definition; our RGB averaging/border policy is a recorded evaluation choice.
