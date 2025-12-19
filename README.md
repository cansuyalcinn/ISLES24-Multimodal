# ISLES24-Multimodal
Multimodal tabular data + vision experiments in the ISLES24 challenge dataset. 

1. 3D UNet (plain model)
- Inputs 5 images
+ Output: binary segmentation

2. 3D UNet (plain model + concatenated tabular data)
- Inputs: 5 images: CTA /MTT /TMAX /CBF / CBV + Clinical data (Excel file - just normalized z-score)

3. 3D UNet + DAFT clinical data
- Inputs: 5 images: CTA /MTT /TMAX /CBF / CBV + Clinical data (Excel file)

Dealing with missing data:

Apply ‘Knockout’

- Randomly drop clinical features while training
- Assign a fixed value (placeholder, e.g., -10) to missing data
