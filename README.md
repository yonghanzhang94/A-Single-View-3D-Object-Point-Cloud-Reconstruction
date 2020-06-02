# img2pointcloud
This repository contains the source codes for the paper 3D-ReConstnet A Single-View 3D-Object Point Cloud Reconstruction Network.
Accepted at IEEE Access. For more details, please see this link:<br>
https://ieeexplore.ieee.org/document/9086481?source=authoralert<br>
and please cite our paper:<br>
B. Li, Y. Zhang, B. Zhao and H. Shao, "3D-ReConstnet: A Single-View 3D-Object Point Cloud Reconstruction Network," in IEEE Access, vol. 8, pp. 83782-83790, 2020, doi: 10.1109/ACCESS.2020.2992554.<br>
![prediction example](https://github.com/yonghanzhang94/img2pointcloud/blob/master/figure3.png)
## Dataset
### ShapeNet
We train and validate our model on the ShapeNet dataset. We use the rendered images from the dataset provided by 3d-r2n2, which consists of 13 object categories. For generating the ground truth point clouds, we sample points on the corresponding object meshes from ShapeNet. We use the dataset split provided by r2n2 in all the experiments. Data download links are provided below: <br>
Rendered Images (~12.3 GB): http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz <br>
ShapeNet pointclouds (~2.8 GB): https://drive.google.com/open?id=1cfoe521iTgcB_7-g_98GYAqO553W8Y0g <br>
ShapeNet train/val split: https://drive.google.com/open?id=10FR-2Lbn55POB1y47MJ12euvobi6mgtc <br>
Download each of the folders, extract them and move them into data/shapenet/.
The folder structure should now look like this:<br>
--data/shapenet/ <br>
&nbsp;&nbsp;--ShapeNetRendering/ <br>
&nbsp;&nbsp;--ShapeNet_pointclouds/ <br>
&nbsp;&nbsp;--splits/ <br>
### Pix3D
We evaluate the generalization capability of our model by testing it on the real-world pix3d dataset. For the ground truth point clouds, we sample 1024 points on the provided meshes. Data download links are provided below: <br>
Pix3D dataset (~20 GB): Follow the instructions in https://github.com/xingyuansun/pix3d <br>
Pix3D pointclouds (~13 MB): https://drive.google.com/open?id=1RZakyBu9lPbG85SyconBn4sR8r2faInV <br>
Download each of the folders, extract them and move them into data/pix3d/.
The folder structure should now look like this:<br>
--data/pix3d/ <br>
&nbsp;&nbsp;--img_cleaned_input/<br>
&nbsp;&nbsp;--img/ <br>
&nbsp;&nbsp;--mask/ <br>
&nbsp;&nbsp;--model/ <br>
&nbsp;&nbsp;--pix3d_pointclouds/ <br>
&nbsp;&nbsp;--pix3d.json <br>
## Usage
Install TensorFlow. We recommend version 1.13 so that the additional TensorFlow ops can be compiled. The code provided has been tested with Python 3.6, TensorFlow 1.13, Pytorch 1.0.0(Lower version can't run) and CUDA 10.0. The following steps need to be performed to run the codes given in this repository:
1.Clone the repository:
```shell
git clone https://github.com/yonghanzhang94/img2pointcloud.git
cd img2pointcloud
```
2.Tensorflow ops for losses (Chamfer and EMD) as well as for point cloud visualization need to be compiled. Run the makefile as given below. (Note that the the nvcc, cudalib, and tensorflow paths inside the makefile need to be updated to point to the locations on your machine):
```shell
make
```
## Training
To train model, run:
```shell
python trainG.py
```
To train model with ambiguous, run:
```shell
python trainG_variety.py
```
## Evaluation
### ShapeNet
For computing the Chamfer and EMD metrics reported in the paper (all 13 categories), run:
```shell
python test_shapenet.py
```
### Pix3D
For computing the Chamfer and EMD metrics reported in the paper (3 categories) for the real-world Pix3D dataset, run:
```shell
python test_pix3d.py
```
