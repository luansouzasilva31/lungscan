import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import SimpleITK as sitk

# LVS: Lung Vessel Segmentation


def extract_nii(basepath: str, savepath: str):
    # Convertendo demais amostras para .nii
    for folder in [
        'ct_scans',
        'masks'
    ]:
        print('Folder: ', folder)
        folder_path = Path(basepath) / folder

        fnames = [f for f in os.listdir(folder_path) if f.endswith('.mhd')]

        for fname in tqdm(fnames, desc=f'Converting {folder}'):
            fpath = folder_path / fname
            image = sitk.ReadImage(fpath)

            if folder == 'masks':
                spath = Path(savepath) / ('lung_' + folder)
            else:
                spath = Path(savepath) / folder

            if not os.path.exists(spath):
                os.makedirs(spath)

            try:
                sitk.WriteImage(image, (spath / fname).with_suffix('.nii.gz'))

            except Exception:
                print('Inconsistent: ', fname)


def check_pairs(data_path: str):
    # Checando se há pares de arquivos .nii.gz
    ct_fnames = [f for f in os.listdir(Path(data_path) / 'ct_scans')
                 if f.endswith('.nii.gz')]
    mk_fnames = [f for f in os.listdir(Path(data_path) / 'lung_masks')
                 if f.endswith('.nii.gz')]

    for ct_fname in ct_fnames:
        if ct_fname not in mk_fnames:
            raise ValueError(f'{ct_fname} not found in lung_masks!')

    for mk_fname in mk_fnames:
        if mk_fname not in ct_fnames:
            raise ValueError(f'{mk_fname} not found in ct_scans!')

    print('All good!')

    return


def analyse_shape(data_path: str):
    fnames = [f for f in os.listdir(Path(data_path) / 'ct_scans')
              if f.endswith('.nii.gz')]

    for fname in fnames:
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
        print('{} >>> ct:{}, mask:{} ... '.format(
            fname, ct_scan.GetSize(), mask.GetSize()), end='')
        if not is_consistent:
            print('Inconsistent! Removing CT and Mask ... ', end='')
            os.remove(ct_path)
            os.remove(mk_path)
            print('Done!')

        else:
            print('Good!')


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


def main():
    raw_data_path = 'data/raw/lung_vessel_segmentation'
    processed_data_path = 'data/interim/lung_vessel_segmentation'

    extract_nii(basepath=raw_data_path, savepath=processed_data_path)
    check_pairs(data_path=processed_data_path)
    analyse_shape(data_path=processed_data_path)
    generate_metadata(data_path=processed_data_path)


if __name__ == '__main__':
    main()
