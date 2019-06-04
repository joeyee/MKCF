# Multiple Kernelized Correlation Filters (MKCF) for Extended Object Tracking

![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)   
This is the **python** implementation of the - 
[Multiple Kernelized Correlation Filters (MKCF) for Extended Object Tracking Using X-band Marine Radar Data](https://ieeexplore.ieee.org/document/8718392).

## Requirements
- python - 3.6.5
- opencv-python

## How to use the code
### Step 1
```bash
python Demo_MKCF.py 

```
<!---
### Step2
Use mouse to select the object which needs to be tracked and Press **Enter** to start tracking.
--->

## Demo Video
[demo on Youku](https://v.youku.com/v_show/id_XNDEwNjQ4MzQyOA==.html?spm=a2hzp.8253876.0.0&f=52133551)

[demo on Github](https://github.com/joeyee/MKCF/blob/master/Video/MKCF_for_alice.mp4)



## Introduction
How to track the target of marine radar in the complex electromagnetic environment? [Need a map here]

Traditional filter based methods (such as kalman filter KF) , which are embedded on the marine radar, could not resist the clutter interruption and ship occlusion in the narrow waterway. It reflects the equipment-embedded tracking method can not fit the association bug for target tracking in the heavy clutter environment. More sophisticated methods like extended object tracking (EOT), particle filters based method are tested but not work neither. 
%
Then the visual tracker KCF attracted our attention. Tracking by detecting visual features may solve the association problem. While we implemented KCF on radar object tracking, we found that KCF performs badly in the long-term tracking. It had been addressed in the \citep{Ma15ICCV}. One common approach is to add more sophisticated feature which would maintain the discriminant of the tracker longer.  Since a Radar may track hundreds of objects simultaneously,  tracker designed for Radar should be light computing. The computing resource on marine radar could not afford those expensive features. One day we luckily discovered that relay-tracking of 3 KCF trackers with simple feature (only the intensity)  on the target named [Alice](https://github.com/joeyee/MKCF/blob/master/Video/MKCF_for_alice.mp4) could be successful during her lifetime in the monitoring zone.

That is the insightful idea from the experiment: a KCF is short-term tracker,  but integrate multiple KCFs, we could get a long-term tracker. Therefore we designed the proposed framework of MKCF. We did not build a math model at first, by contrast, we built the framework in favor of efficient numerical computing, which is suitable for automatic initializing and fusing KCF in the radar signal. The proposed framework succeeded to track all the objects in our experiments. It provides a novel approach to  solve the  engineering difficulty, and we do want to share it with the community. 

When I studied the handbook of data fusion \citep{Liggins08SensorFusionBook} for deducing the fusion equations,  I had noted that there was a connection between the maximum likelihood estimation and KCF. The unique circular shifting samples in KCF are similar to the particle samples in particle filters (PF) based visual tracking, if we choose the particle to be dense enough. The element in the Kernel matrix of KCF is in fact a likelihood function in PF, which measures the similarity between the reference and the candidate sample.  If we can explain the single KCF in the view point of maximum likelihood, of course it is natural to fusing multiple of KCF via the maximum likelihood technique.
%
The advantage of KCF is that, once get the Kernel matrix $K_{xx}$ in the reference, she try to learn the pattern $\boldsymbol{\alpha}$ of the matrix. In the later frame, the convolution of the tested Kernel matrix $K_{xz}$ and $\boldsymbol{\alpha}$  produce the response map, saying the probability of a sample to be the true target. Here we can see that the PF and KCF both relies heavily on the reference model of the target which is computed on the initial frame. KCF uses naive running average to update the reference model. It is simple in computing, but can not handle the multiple models of the true target in long-term tracking. Therefore there should be a on-off key (PSR) for updating.  The component of MKCF are independent KCF, which is dynamically initialized on different frame. Consequently, \textbf{MKCF maintains multiple instances of the true target in different time steps. This technique enhances the robustness of the tracker in long-term tracking without using time-consuming visual features. It works in radar signal and  also provides useful experience for the video oriented visual tracking.}





## Reference:
[1] [Multiple Kernelized Correlation Filters (MKCF) for Extended Object Tracking Using X-band Marine Radar Data](https://ieeexplore.ieee.org/document/8718392).
