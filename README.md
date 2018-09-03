# Depth-denoising-Kinect
A captured depth map is a degraded version of the underlying ground truth. Denote by u the depth map
of a scene. Our aim is to obtain a high quality joint 3-D reconstruction from the low quality depth maps
in a consolidated framework that is robust to surface curvatures. We assume the sensors are subject to
Gaussian noise and optimize over the depth values of the sensor. It is an iterative process that smoothens
the 2D representation of the scene. The observation model assumes the depth maps to be fronto-parallel
planes and the sensors are calibrated intrinsically. Another assumption we make is that it is a piecewise
planar model.
