# RP-PCN: Rapeseed Population Point Cloud Completion Network

This repository contains the **PyTorch implementation** of **RP-PCN**, a point cloud completion network designed for reconstructing the complete canopy structure of field-grown rapeseed populations. This work extends **PF-Net(CVPR 2020)**, incorporating a **multi-resolution dynamic graph encoder (MRDG) and dynamic graph convolutional feature extractor (DGCFE)** to enhance the reconstruction of complex plant architectures.

For further details, please refer to our paper: [unpublished].

---

## 0) Environment Setup
The code is developed based on **PyTorch 1.0.1** and **Python 3.7.4**, but additional dependencies require a **Python 3.8+ environment**. 

### **Dependencies**
- Python 3.8+
- PyTorch 1.0.1
- Open3D
- COLMAP
- Trimesh
- PyEmbree (for ray-tracing acceleration)

**Installation:**
```
conda create -n rp-pcn python=3.8
conda activate rp-pcn
pip install -r requirements.txt
```

---

## 1) Dataset Preparation

This project generates a **custom rapeseed population point cloud dataset** rather than relying on pre-existing datasets like ShapeNet. 

### **Steps to generate the dataset:**
1. **Extract camera pose information using COLMAP**
   ```
   python COLMAP_batch.py
   ```
   This script processes input images to estimate their poses using COLMAP.

2. **Perform 3D reconstruction using Instant-NGP**
   - Run **Instant-NGP** to reconstruct individual rapeseed plants.
   - Export the reconstructed **mesh files**.

3. **Generate the training dataset**
   ```
   python create_dataset.py
   ```
   - This step creates the training dataset using **exported mesh files**.
   - **Note**: The dataset preparation step requires **Python 3.8+** with **Trimesh** and **PyEmbree** installed.

⚠ **Important:** Sample dataset files are provided in this repository, but full datasets must be generated using the above pipeline.

---

## 2) Training RP-PCN
To train RP-PCN on the generated dataset, run:
```
python Train_RPPCN.py
```
**Hyperparameters:**
- `crop_point_num`: Controls the number of missing points.
- `point_scales_list`: Controls different input resolutions.
- `D_choose`: Enables or disables specific modules.

---

## 3) Evaluation
To evaluate the trained model:
```
python evaluate.py
```
This script calculates **Chamfer Distance (CD)** and visualizes the reconstruction results.

To visualize the results:
```
python visualize_results.py
```
This script generates **point cloud reconstructions** and stores them in `results/`.

---

## 4) Visualization of Point Cloud Files
To visualize point clouds, we recommend using **CloudCompare** or **MeshLab**.

For `.csv` format point clouds:
```
python Test_csv.py
```
Modify `infile` and `infile_real` in the script to select different test samples.

---

## 5) Example Files and Additional Data
This repository includes:
- A few **sample input and output point clouds** for testing.
- Scripts to **generate missing point clouds and complete them using RP-PCN**.

For full datasets and additional training data, please refer to **[GitHub Dataset Repository (TBD)]**.

---

If you have any questions regarding the code or dataset, please contact **[ziyue_guo@zju.edu.cn]**. 

Happy Coding! 🚀
