# inference.py imports
import glob 
from pathlib import Path
import SimpleITK
import json
import subprocess
import os
import sys
from os.path import join
import shutil
import SimpleITK as sitk
# import preprocessing  # merged files, no need to import

import argparse

# preprocessing.py imports
import nibabel as nib
import numpy as np
from skimage import exposure
from concurrent.futures import ProcessPoolExecutor


'''
Preprocessing code for KurtLab ISLES'24 submission, reformatted for command line execution
Excerpted from https://github.com/KurtLabUW/ISLES2024/tree/main

Images in args.input_dir are first renamed to match the nnUNet format using nnunet_dataset_conversion.
Then  they are preprocessed in line 142 with preprocessed = run_preprocessing(data, args.preprocessed_dir)
This preprocesses all cases in args.input_dir, indexing by their identifier (e.g for files like 'isles0000_0001.nii', the identifier is '0000')

NOTE: for consistency with the original Docker submission syntax, absolute paths, like "/tmp/", were renamed to local paths, like "tmp/". All of these changes are noted as comments

Arguments:

- args.input_dir: contains the images being preprocessed, structured as in the ISLES24 submission. In this example this looks like:

    └── args.input_dir/
        └── images/
            ├── preprocessed-cbf-map/
            │   ├── preprocessed-cbf-map.mha
            ├── preprocessed-ct-angiography/
            │   ├── preprocessed-ct-angiography.mha
            ├── preprocessed-cbv-map/
            │   ├── preprocessed-cbv-map.mha
            ├── preprocessed-tmax-map/
            │   ├── preprocessed-tmax-map.mha
            └── preprocessed-mtt-map/
                ├── preprocessed-mtt-map.mha

- args.preprocessed_dir: location where preprocessed images are to be saved. Example

    └── args.preprocessed_dir/
        ├── isles0000_0000.nii.gz # CBF
        ├── isles0000_0001.nii.gz # CBV
        ├── isles0000_0002.nii.gz # MTT
        ├── isles0000_0003.nii.gz # TMax
        └── isles0000_0004.nii.gz # CTA
'''


## START functions from preprocessing.py
def apply_histogram_equalization_custom_range(input_file, output_file, min_intensity, max_intensity):
    img = nib.load(input_file)
    data = img.get_fdata()

    # Clip the data to the custom intensity range
    data_clipped = np.clip(data, min_intensity, max_intensity)
    data_clipped[data == 0] = 0
    # Normalize the data to [0, 1] range
    data_clipped[data_clipped > 0.0001] -= min_intensity
    data_normalized = data_clipped / (max_intensity - min_intensity)

    # Apply 3D histogram equalization
    equalized_data = exposure.equalize_hist(data_normalized, mask=(data_normalized > 0.0001))
    
    equalized_data[data_normalized < 0.0001] = 0
     
    # Save the result as a new NIfTI file
    equalized_img = nib.Nifti1Image(equalized_data, img.affine, img.header)
    nib.save(equalized_img, output_file)
    print(f"Saved equalized image to {output_file}")

def process_training_case(case_identifier, input_dir, output_dir, intensity_ranges):
    case_files = sorted(glob.glob(os.path.join(input_dir, f"{case_identifier}_*.nii.gz")))
    # Process each channel based on its 4-digit identifier
    for input_file in case_files:
        channel_id = input_file.split('_')[-1].split('.')[0]
        if channel_id in intensity_ranges:
            min_intensity, max_intensity = intensity_ranges[channel_id]
            output_file = os.path.join(output_dir, f"{case_identifier}_{channel_id}.nii.gz")
            apply_histogram_equalization_custom_range(input_file, output_file, min_intensity, max_intensity)
            
        else:
            # If the channel is not in intensity_ranges, simply copy it
            output_file = os.path.join(output_dir, f"{case_identifier}_{channel_id}.nii.gz")
            shutil.copy(input_file, output_file)
            print(f"Copied {input_file} to {output_file}")

def process_all_cases(input_dir, output_dir, intensity_ranges):
    case_identifiers = {os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(input_dir, "*_*.nii.gz"))}
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_training_case, case_identifier, input_dir, output_dir, intensity_ranges)
                   for case_identifier in case_identifiers]
        for future in futures:
            future.result()  # This will raise any exceptions encountered during processing

def run_preprocessing(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Define custom intensity ranges for each modality/channel
    intensity_ranges = {
        '0000': (0, 35),   # Example range for channel 0000 (e.g., T1)
        '0001': (0, 10), # Example range for channel 0001 (e.g., T2)
        '0002': (0, 20),  # Example range for channel 0002
        '0003': (0, 7),
        '0004': (0, 90),
    }

    process_all_cases(input_dir, output_dir, intensity_ranges)
    print("Processing complete.")
    return output_dir

## END functions from preprocessing.py




## START functions from inference.py
def move_file(source, dest):
    img = sitk.ReadImage(source)
    sitk.WriteImage(img, dest)

def nnunet_dataset_conversion(data_path):
    out_base = "tmp/raw" # changed from "/tmp/raw"
    
    if not os.path.exists(out_base):
        os.makedirs(out_base)
    move_file(glob.glob(str(join(data_path, "preprocessed-cbf-map", "*.mha")))[0], "tmp/raw/isles0000_0000.nii.gz") # changed from "/tmp/raw"...
    move_file(glob.glob(str(join(data_path, "preprocessed-cbv-map", "*.mha")))[0], "tmp/raw/isles0000_0001.nii.gz") # changed from "/tmp/raw"...
    move_file(glob.glob(str(join(data_path, "preprocessed-mtt-map", "*.mha")))[0], "tmp/raw/isles0000_0002.nii.gz") # changed from "/tmp/raw"...
    move_file(glob.glob(str(join(data_path, "preprocessed-tmax-map", "*.mha")))[0], "tmp/raw/isles0000_0003.nii.gz") # changed from "/tmp/raw"...
    move_file(glob.glob(str(join(data_path, "preprocessed-CT-angiography", "*.mha")))[0], "tmp/raw/isles0000_0004.nii.gz") # changed from "/tmp/raw"...
    
    return "tmp/raw" # changed from "/tmp/raw"

## END functions from inference.py




def main(args):
    data = nnunet_dataset_conversion(Path(args.input_dir) / "images")
    preprocessed = run_preprocessing(data, args.preprocessed_dir)
    ### nnunet inference goes here...

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", help="Path to the top-level directory containing the ISLES24-style input. ", type=str, default="input") # changed from "/input")
    p.add_argument("--preprocessed_dir", help="Directory where preprocessed nnUNet-ready files will be saved.", type=str, default="tmp/preprocessed") # changed from "/tmp/preprocessed"

    args = p.parse_args()
    main(args)

    '''
    Command:
        python isles_preprocess_kurtlab.py 

    Example output:
        Saved equalized image to tmp/preprocessed/isles0000_0000.nii.gz
        Saved equalized image to tmp/preprocessed/isles0000_0001.nii.gz
        Saved equalized image to tmp/preprocessed/isles0000_0002.nii.gz
        Saved equalized image to tmp/preprocessed/isles0000_0003.nii.gz
        Saved equalized image to tmp/preprocessed/isles0000_0004.nii.gz
        Processing complete.
    '''