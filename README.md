# Multiple Kernelized Correlation Filters (MKCF) for Extended Object Tracking

![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)   
This is the **python** implementation of the - 
[Multiple Kernelized Correlation Filters (MKCF) for Extended Object Tracking Using X-band Marine Radar Data](https://ieeexplore.ieee.org/document/8718392).

[Source paper](https://github.com/joeyee/MKCF/blob/master/MKCF_SourcePaper_SingleColumn.pdf) of the preprint version with supplementary materials could be download at Github.

## Requirements
- python - 3.6.5
- opencv-python

## How to use the code

### Step 1
Download the raw radar data from the [Net disk of Baidu](https://pan.baidu.com/s/1GJ6JR9jfYVLR7OfRLtkQlg) with PWD (yu67) 
<!---
![Wechat_download](https://github.com/joeyee/MKCF/blob/master/images/baidu_qrcode.jpeg), 
-->
or [Google drive]()
```Python
#following code is to test and view the raw radar data
fdata_obj = open('path of the downloaded radar data file', 'rb')
while True:
    #Show the data with frame by frame
    frame = np.fromfile(fdata_obj, 'float64', 600 * 2048)
    if frame.size == 0:
       break
    uframe = (frame * 255).astype(np.uint8) # convert double to unit for displaying with opencv
    #view the raw data in the polar-coordinates
    cv2.imshow('polar', uframe)
    #view in the cartesian-coordinates
    dispmat = corcon.polar2disp_njit(frame, np.array([]))
    cv2.imshow('Descartes', dispmat)
    cv2.waitKey(1)
```
### Step 2
```bash
python Demo_MKCF_approx.py 

```
<!---
### Step2
Use mouse to select the object which needs to be tracked and Press **Enter** to start tracking.
--->

## Demo Video
[Performance Comparision](https://v.youku.com/v_show/id_XNDEwNjQ4MzQyOA==.html?spm=a2hzp.8253876.0.0&f=52133551)

[MKCF Demo](https://v.youku.com/v_show/id_XNDEwNjQ4NDE5Mg==.html?spm=a2h0j.11185381.listitem_page1.5!2~A&&f=52133551)



## Introduction
How to track the target of marine radar in the complex electromagnetic environment?
![radar_scene](https://github.com/joeyee/MKCF/blob/master/images/radar_scene.png)
 <sub>Heat map of the radar image captured in an inner river site. Ships, buoys in the river, bank of the river and railway bridge across the river show strong intensities in deep red color. When the ship is approaching the radar site (top center of the image), multi-path clutter makes the shape of the ship irregular and causes interruption to the neighbor ship. The radar image is in *range*-*azimuth* format (the vertical axis maps the *range*} units, and the horizontal denotes the *azimuth*)</sub>

Traditional filter based methods (such as kalman filter KF) , which are embedded on the marine radar, could not resist the clutter interruption and ship occlusion in the narrow waterway. It reflects the equipment-embedded tracking method can not fit the association bug for target tracking in the heavy clutter environment. More sophisticated methods like extended object tracking (EOT), particle filters based method are tested but not work neither. 

Then the visual tracker KCF[[Henriques et al., 2012]](http://www.robots.ox.ac.uk/~joao/publications/henriques_eccv2012.pdf) attracted our attention. Tracking by detecting visual features may solve the association problem. While we implemented KCF on radar object tracking, we found that KCF performs badly in the long-term tracking. It had been addressed in the [[Ma et al., 2015]](https://ieeexplore.ieee.org/document/7410709/). One common approach is to add more sophisticated feature which would maintain the discriminant of the tracker longer.  Since a Radar may track hundreds of objects simultaneously,  tracker designed for Radar should be light computing. The computing resource on marine radar could not afford those expensive features. One day we luckily discovered that relay-tracking of 3 KCF trackers with simple feature (only the intensity)  on the target named [Alice](https://v.youku.com/v_show/id_XNDEwNjQ4NDE5Mg==.html?spm=a2h0j.11185381.listitem_page1.5!2~A&&f=52133551) could be successful during her lifetime in the monitoring zone.

That is the insightful idea from the experiment: a KCF is short-term tracker,  but integrate multiple KCFs, we could get a long-term tracker. Therefore we designed the proposed framework of MKCF. We did not build a math model at first, by contrast, we built the framework in favor of efficient numerical computing, which is suitable for automatic initializing and fusing KCF in the radar signal. The proposed framework succeeded to track all the objects in our experiments. It provides a novel approach to  solve the  engineering difficulty, and we do want to share it with the community. 

When I studied the handbook of data fusion [Liggins et al., 2008] for deducing the fusion equations,  I had noted that there was a connection between the maximum likelihood estimation and KCF. The unique circular shifting samples in KCF are similar to the particle samples in particle filters (PF) based visual tracking, if we choose the particle to be dense enough. The element in the Kernel matrix of KCF is in fact a likelihood function in PF, which measures the similarity between the reference and the candidate sample.  If we can explain the single KCF in the view point of maximum likelihood, of course it is natural to fusing multiple of KCF via the maximum likelihood technique.

The advantage of KCF is that, once get the Kernel matrix ![Kxx](https://latex.codecogs.com/svg.latex?K_{xx}) in the reference, she try to learn the pattern ![alpha](https://latex.codecogs.com/svg.latex?\alpha) of the matrix. In the later frame, the convolution of the tested Kernel matrix ![Kxz](https://latex.codecogs.com/svg.latex?K_{xz}) and ![alpha](https://latex.codecogs.com/svg.latex?\alpha) produce the response map, saying the probability of a sample to be the true target. Here we can see that the PF and KCF both relies heavily on the reference model of the target which is computed on the initial frame. KCF uses naive running average to update the reference model. It is simple in computing, but can not handle the multiple models of the true target in long-term tracking. Therefore there should be a on-off key (PSR) for updating.  The component of MKCF are independent KCF, which is dynamically initialized on different frame. Consequently, **MKCF maintains multiple instances of the true target in different time steps. This technique enhances the robustness of the tracker in long-term tracking without using time-consuming visual features. It works in radar signal and  also provides useful experience for the video oriented visual tracking.**

<!---
## Diagram of the MKCF
![Sequential](https://github.com/joeyee/MKCF/blob/master/images/Diagram_MKCF.png)

![oneStep](https://github.com/joeyee/MKCF/blob/master/images/diagram_one_timestep.png)
-->


## Reference:
[1] [Multiple Kernelized Correlation Filters (MKCF) for Extended Object Tracking Using X-band Marine Radar Data](https://ieeexplore.ieee.org/document/8718392).

[[Henriques et al., 2012]](http://www.robots.ox.ac.uk/~joao/publications/henriques_eccv2012.pdf)J. Henriques, R. Caseiro, P. Martins, and J. Batista. Exploiting the circulant structure of tracking-by-detection with kernels. In Proceedings of the European Conference on Computer Vision, 2012.


[[Ma et al., 2015]](https://ieeexplore.ieee.org/document/7410709/) Chao Ma, Jia-Bin Huang, Xiaokang Yang, and Ming-Hsuan Yang. Hierarchical convolutional features for visual tracking. In Proceedings of the IEEE International Conference on Computer Vision, pages 3074–3082, 2015. 1

[Liggins et al., 2008]Martin E. Liggins, David L. Hall, and James Llinas. Handbook of multisensor data fusion: theory and practice. CRC Press, 2nd edition, 2008. 2, 19, 21

<!---
<dl>
<script type="text/javascript" id="clstr_globe" src="//cdn.clustrmaps.com/globe.js?d=5B4XJjSp3_gxkzPck_Uh7bPH2hr1JEGySA5tIbewhpQ"></script>
</dl>
--->


