# ISLES24-Multimodal
Multimodal tabular data + vision experiments in ISLES24 challenge dataset. 

1. 3D UNet (plain model)
- Inputs 6 images
+ Output: binary segmentation

2. 3D UNet (plain model + concatenated tabular data)
- Inputs: 6 images: CTA / NCCT /MTT /TMAX /CBF / CBV + Clinical data (excel file - just normalized z-scpre)

3. 3D UNet + DAFT clinical data
- Inputs: 6 images: CTA / NCCT /MTT /TMAX /CBF / CBV + Clinical data (excel file)

Dealing with missing data:

Apply ‘Knockout’

- Randomly drop clinical features while training
- Assign a fix value (placeholder, e.g. -10) to missing data
