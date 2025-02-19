# NeuralDeformableModels
This is the project page of the ICCV 2023 paper NeuralDeformableModels [[ICCV_2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Ye_Neural_Deformable_Models_for_3D_Bi-Ventricular_Heart_Shape_Reconstruction_and_ICCV_2023_paper.pdf)][[arXiv](https://arxiv.org/pdf/2307.07693)].

## Neural Deformable Models for 3D Bi-Ventricular Heart Shape Reconstruction and Modeling from 2D Sparse Cardiac Magnetic Resonance Imaging.
We proposed a novel neural deformable model (NDM) targeting at the reconstruction and modeling of 3D bi-ventricular shape of the heart from 2D sparse cardiac magnetic resonance (CMR) imaging data.
<div align=center><img width="820" height="340" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/NDMs.png"/></div>

## Demo 1
The following videoes show a full cardiac cycle of left ventricle myocardium wall shapes reconstructed from a normal subject cine MRI. 
<div align=center><img width="95" height="115.5" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/Normal/LAX/0_2.gif"/><img width="95" height="115.5" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/Normal/LAX/0_5.gif"/><img width="95" height="115.5" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/Normal/LAX/0_4.gif"/><img width="92" height="115.5" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/Normal/SAX/0_3.gif"/><img width="92" height="115.5" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/Normal/SAX/0_6.gif"/><img width="92" height="115.5" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/Normal/SAX/0_9.gif"/></div>
<div align=center><img width="400" height="400" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/Normal/LV_wall_dynamics4_0.12.gif"/></div>
<div align=center><img width="820" height="178.62" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/Normal/NHC008.png"/></div>

## Demo 2
The following videoes show a full cardiac cycle of left ventricle myocardium wall shapes reconstructed from a diseased subject cine MRI. 
<div align=center><img width="250" height="237.4" src="https://github.com/DeepTag/VolumetricNeuralDeformableModels/blob/main/Simulation_Example/Apparent_Lagrangian_Motion/SAX/apparent_motion_sequence_sax_7.gif"/><img width="250" height="237.4" src="https://github.com/DeepTag/VolumetricNeuralDeformableModels/blob/main/Simulation_Example/Apparent_Lagrangian_Motion/SAX/apparent_motion_sequence_sax_5.gif"/><img width="250" height="237.4" src="https://github.com/DeepTag/VolumetricNeuralDeformableModels/blob/main/Simulation_Example/Apparent_Lagrangian_Motion/SAX/apparent_motion_sequence_sax_3.gif"/></div>
<div align=center><img width="250" height="238.6" src="https://github.com/DeepTag/VolumetricNeuralDeformableModels/blob/main/Simulation_Example/Apparent_Lagrangian_Motion/LAX/apparent_motion_sequence_lax_0.gif"/><img width="250" height="238.6" src="https://github.com/DeepTag/VolumetricNeuralDeformableModels/blob/main/Simulation_Example/Apparent_Lagrangian_Motion/LAX/apparent_motion_sequence_lax_1.gif"/><img width="250" height="238.6" src="https://github.com/DeepTag/VolumetricNeuralDeformableModels/blob/main/Simulation_Example/Apparent_Lagrangian_Motion/LAX/apparent_motion_sequence_lax_2.gif"/></div>

## Demo 3
The following video shows a full cardiac cycle of left ventricle myocardium wall shapes reconstructed from multiplanar [tagged-MRI](https://github.com/DeepTag/cardiac_tagging_motion_estimation) cine with our Neural Deformable Models.  
<div align=center><img width="400" height="400" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/LV_wall_dynamics.gif"/></div>
<div align=center><img width="820" height="178.62" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/cardiac_cycle_LVs.png"/></div>

## Demo 4
The following video shows a full cardiac cycle of left ventricle myocardium wall meshes reconstructed from multi-planar [tagged-MRI](https://github.com/DeepTag/cardiac_tagging_motion_estimation) cine with our Neural Deformable Models. Note that the twisting motion has been successfully recovered. 
<div align=center><img width="400" height="400" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/LV_wall_dynamics_twist.gif"/></div>
