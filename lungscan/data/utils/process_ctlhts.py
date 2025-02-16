import os
import glob
import shutil
import dicom2nifti
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import SimpleITK as sitk
from termcolor import colored

# CTLHTS: CT Lung and Heart and Trachea Segmentation


def extract_nii(basepath: str, savepath: str):
    # CT Scans
    ct_dir = os.path.join(basepath, 'nrrd_noisy', 'nrrd_noisy')
    fnames = os.listdir(ct_dir)

    for fname in tqdm(fnames, desc='Converting CT Scans'):
        from_path = Path(ct_dir) / fname
        image = sitk.ReadImage(from_path)

        # some images incorrectly have empty space
        new_fname = fname.replace(' ', '')
        new_fname = new_fname.replace('_noisy.nrrd', '.nii.gz')
        to_path = Path(savepath) / 'ct_scans' / new_fname

        if not os.path.exists(to_path.parent):
            os.makedirs(to_path.parent)

        sitk.WriteImage(image, to_path)

    # Masks
    mask_dir = os.path.join(basepath, 'masks', 'nrrd_lung', 'nrrd_lung')
    fnames = os.listdir(mask_dir)

    for fname in tqdm(fnames, desc='Converting Masks'):
        from_path = Path(mask_dir) / fname
        image = sitk.ReadImage(from_path)

        # some images incorrectly have empty space
        new_fname = fname.replace(' ', '')
        new_fname = new_fname.replace('_lung.nrrd', '.nii.gz')
        to_path = Path(savepath) / 'lung_masks' / new_fname

        if not os.path.exists(to_path.parent):
            os.makedirs(to_path.parent)

        sitk.WriteImage(image, to_path)

    return


def check_pairs(data_path: str):
    # Checando se há pares de arquivos .nii.gz
    ct_fnames = [f for f in os.listdir(Path(data_path) / 'ct_scans')
                 if f.endswith('.nii.gz')]
    mk_fnames = [f for f in os.listdir(Path(data_path) / 'lung_masks')
                 if f.endswith('.nii.gz')]

    print('Check pairs ... CT Scan -> Lung Mask')
    for ct_fname in ct_fnames:
        if ct_fname not in mk_fnames:
            print(f'{ct_fname} not found in lung_masks. Deleting...')
            os.remove(Path(data_path) / 'ct_scans' / ct_fname)

    print('Check pairs ... Lung Mask -> CT Scan')
    for mk_fname in mk_fnames:
        if mk_fname not in ct_fnames:
            print(f'{mk_fname} not found in ct_scans. Deleting...')
            os.remove(Path(data_path) / 'lung_masks' / mk_fname)

    return


def analyse_shape(data_path: str):
    fnames = [f for f in os.listdir(Path(data_path) / 'ct_scans')
              if f.endswith('.nii.gz')]

    for i, fname in enumerate(fnames):
        ct_path = Path(data_path) / 'ct_scans' / fname
        mk_path = Path(data_path) / 'lung_masks' / fname

        # Importando ct_scan e mask
        ct_scan = sitk.ReadImage(ct_path)
        mask = sitk.ReadImage(mk_path)

        # Verificando se há amostras inconsistentes
        is_consistent = (
            ct_scan.GetSize() == mask.GetSize()
            # and ct_scan.GetSize()[0] == ct_scan.GetSize()[1]
        )

        # Deletando amostras inconsistentes
        print('[{}/{}] {} >>> ct:{}, mask:{} ... '.format(
            i + 1, len(fnames), fname, ct_scan.GetSize(), mask.GetSize()),
            end='')
        if not is_consistent:
            print('Inconsistent! Removing CT and Mask ... ', end='')
            os.remove(ct_path)
            os.remove(mk_path)
            print('Done!')

        else:
            print('Good!')

    return


def generate_metadata(data_path: str):
    fnames = [f for f in os.listdir(Path(data_path) / 'ct_scans')
              if f.endswith('.nii.gz')]

    metadata = list()

    for fname in tqdm(fnames, desc='Generating metadata'):
        info = dict()

        ct_path = Path(data_path) / 'ct_scans' / fname
        ct_scan = sitk.ReadImage(ct_path)
        shape = ct_scan.GetSize()

        info['file_name'] = fname
        info['ct_path'] = os.path.join('ct_scans', fname)
        info['lung_mask_path'] = os.path.join('lung_masks', fname)
        info['shape'] = shape

        metadata.append(info)

    metadata = pd.DataFrame(metadata)
    metadata.to_csv(Path(data_path) / 'metadata.csv', index=False)

    return


if __name__ == '__main__':
    raw_data_path = 'data/raw/ct_lung_and_heart_and_trachea_segmentation'
    processed_data_path = 'data/interim/ct_lung_and_heart_and_trachea_segmentation'

    # extract_nii(basepath=raw_data_path, savepath=processed_data_path)
    # check_pairs(data_path=processed_data_path)
    # analyse_shape(data_path=processed_data_path)
    generate_metadata(data_path=processed_data_path)
