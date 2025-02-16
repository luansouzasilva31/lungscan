import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk
from pathlib import Path
from dcmrtstruct2nii import dcmrtstruct2nii

# LCTSC: Lung CT Segmentation Challenge


def extract_nii(basepath: str, savepath: str):
    metadata = pd.read_csv(os.path.join(basepath, 'metadata.csv'))
    fpaths = metadata['File Location'].tolist()

    for fpath in tqdm(fpaths, desc='Extracting .nii'):
        rs_path = Path(basepath) / fpath
        rs_name = rs_path.name

        rs2_name = os.listdir(rs_path.parent)  # list
        rs2_name.remove(rs_name)  # list
        rs2_name = rs2_name[0]  # str

        # the metadata has the path to the rstruct and dicom files.
        # we only need the rstruct. If the folder is dicom, we skip.
        if os.path.isfile(rs_path / '1-1.dcm'):
            dcmrtstruct2nii(
                rs_path / '1-1.dcm',
                rs_path.parent / rs2_name,
                Path(savepath) / rs_path.parents[1].name
            )

    return


def build_lung_mask(data_path: str):
    fnames = os.listdir(data_path)

    for fname in tqdm(fnames, desc='Building lung mask'):
        try:
            mask_lung_r = sitk.ReadImage(
                Path(data_path) / fname / 'mask_Lung_R.nii.gz')
        except Exception:
            print(f'{fname} >>> mask_Lung_R not found!')
            continue

        try:
            mask_lung_l = sitk.ReadImage(
                Path(data_path) / fname / 'mask_Lung_L.nii.gz')
        except Exception:
            print(f'{fname} >>> mask_Lung_L not found!')
            continue

        mask_lung_r = sitk.GetArrayFromImage(mask_lung_r)
        mask_lung_l = sitk.GetArrayFromImage(mask_lung_l)

        mask_lung = np.sum((mask_lung_r, mask_lung_l), axis=0)
        mask_lung = sitk.GetImageFromArray(mask_lung)

        sitk.WriteImage(
            mask_lung, Path(data_path) / fname / 'mask_Lung.nii.gz')

    return


def check_pairs(data_path: str):
    print('Checking pairs...', end=' ')

    fnames = os.listdir(data_path)
    for fname in fnames:
        if not os.path.isfile(Path(data_path) / fname / 'image.nii.gz'):
            raise ValueError(f'{fname} >>> image.nii.gz not found!')

        if not os.path.isfile(Path(data_path) / fname / 'mask_Lung_L.nii.gz'):
            raise ValueError(f'{fname} >>> mask_Lung_L.nii.gz not found!')

        if not os.path.isfile(Path(data_path) / fname / 'mask_Lung_R.nii.gz'):
            raise ValueError(f'{fname} >>> mask_Lung_R.nii.gz not found!')

        if not os.path.isfile(Path(data_path) / fname / 'mask_Lung.nii.gz'):
            raise ValueError(f'{fname} >>> mask_Lung.nii.gz not found!')

    print('All good!')

    return


def analyse_shape(data_path: str):
    fnames = os.listdir(data_path)

    for i, fname in enumerate(fnames):
        ct_path = Path(data_path) / fname / 'image.nii.gz'
        mk_path = Path(data_path) / fname / 'mask_Lung.nii.gz'

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
            shutil.rmtree(Path(data_path) / fname)
            print('Done!')

        else:
            print('Good!')

    return


def restructure_data(data_path: str):
    fnames = os.listdir(data_path)

    for fname in tqdm(fnames, desc='Restructuring data'):
        from_dir = Path(data_path) / fname

        to_dir = Path(data_path) / 'ct_scans'
        if not os.path.exists(to_dir):
            os.makedirs(to_dir)

        shutil.move(from_dir / 'image.nii.gz', to_dir / f'{fname}.nii.gz')

        to_dir = Path(data_path) / 'lung_masks'
        if not os.path.exists(to_dir):
            os.makedirs(to_dir)

        shutil.move(from_dir / 'mask_Lung.nii.gz', to_dir / f'{fname}.nii.gz')

        # remove old directory
        shutil.rmtree(Path(data_path) / fname)

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
        info['ct_scan'] = os.path.join('ct_scans', fname)
        info['lung_mask_path'] = os.path.join('lung_masks', fname)
        info['shape'] = shape

        metadata.append(info)

    metadata = pd.DataFrame(metadata)
    metadata.to_csv(Path(data_path) / 'metadata.csv', index=False)

    return metadata


def main():
    raw_data_path = 'data/raw/lctsc'
    processed_data_path = 'data/interim/lctsc'

    extract_nii(basepath=raw_data_path, savepath=processed_data_path)
    build_lung_mask(data_path=processed_data_path)
    check_pairs(data_path=processed_data_path)
    analyse_shape(data_path=processed_data_path)
    restructure_data(data_path=processed_data_path)
    generate_metadata(data_path=processed_data_path)


if __name__ == '__main__':
    main()
