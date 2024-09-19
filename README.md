# NeuralDeformableModels
This is the project page of the ICCV 2023 paper NeuralDeformableModels [[ICCV_2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Ye_Neural_Deformable_Models_for_3D_Bi-Ventricular_Heart_Shape_Reconstruction_and_ICCV_2023_paper.pdf)][[arXiv](https://arxiv.org/pdf/2307.07693)].

## Neural Deformable Models for 3D Bi-Ventricular Heart Shape Reconstruction and Modeling from 2D Sparse Cardiac Magnetic Resonance Imaging.
We proposed a novel neural deformable model (NDM) targeting at the reconstruction and modeling of 3D bi-ventricular shape of the heart from 2D sparse cardiac magnetic resonance (CMR) imaging data.
<div align=center><img width="820" height="340" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/NDMs.png"/></div>

## Demo 1
The following video shows a full cardiac cycle of left ventricle myocardium wall shapes reconstructed from multiplanar [tagged-MRI](https://github.com/DeepTag/cardiac_tagging_motion_estimation) cine with our Neural Deformable Models.  
<div align=center><img width="400" height="400" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/LV_wall_mesh.gif"/></div>
<div align=center><img width="820" height="174.5" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/Cardiac_cycle_with_wall_shapes.png"/></div>

## Demo 2
The following video shows a full cardiac cycle of left ventricle myocardium wall meshes reconstructed from multiplanar [tagged-MRI](https://github.com/DeepTag/cardiac_tagging_motion_estimation) cine with our Neural Deformable Models. Note that the twisting motion has been successfully recovered. 
<div align=center><img width="400" height="400" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/LV_wall_dynamics_twist.gif"/></div>

## Demo 3
The following videoes show the SAX apical/middle/basal view and LAX 2ch/3ch/4ch view apparent motion sequences. 
<div align=center><img width="250" height="250" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/apparent_motion_sequence_sax_apical.gif"/><img width="250" height="250" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/apparent_motion_sequence_sax_middle.gif"/><img width="250" height="250" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/apparent_motion_sequence_sax_basal.gif"/></div>
<div align=center><img width="250" height="250" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/apparent_motion_sequence_lax_2ch.gif"/><img width="250" height="250" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/apparent_motion_sequence_lax_3ch.gif"/><img width="250" height="250" src="https://github.com/DeepTag/NeuralDeformableModels/blob/main/apparent_motion_sequence_lax_4ch.gif"/></div>
