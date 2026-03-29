# FuzzyAlign

FuzzyAlign is a set of scripts for EEG-to-image alignment and reconstruction on TVSD-like data.

This release removes machine-specific absolute paths and supports configurable paths through CLI arguments.

## What Is Included

- EEG-image alignment training: `fuzzyalign.py`
- TVSD preprocessing: `TVSD/preprocessing/`
- DNN feature extraction: `TVSD/dnn_feature_extraction/`
- Reconstruction pipeline: `reconstruction/`
- Reconstruction evaluation metrics: `metrics/`

## Project Structure

```text
FuzzyAlign/
  fuzzyalign.py
  metrics/
    recon_metrics_low.py
    recon_metrics_high.py
  reconstruction/
    image_condition.py
    vae_alignment.py
    image_generation_low.py
    image_generation_high.py
    diffusion_prior.py
    model.py
  TVSD/
    preprocessing/
      preprocessing.py
      find_test100.py
    dnn_feature_extraction/
      feature_maps_clip_h14.py
      feature_maps_vae.py
```

## Environment

Recommended:

- Python 3.10+
- PyTorch
- torchvision
- diffusers
- transformers
- open-clip-torch
- pandas
- numpy
- scipy
- scikit-image
- matplotlib
- einops
- torch-geometric

Install missing dependencies with your preferred package manager.

```bash
pip install -r FuzzyAlign/requirements.txt
```

## Expected Data Layout

The scripts now default to relative paths. One recommended layout is:

```text
<repo_root>/
  FuzzyAlign/
  TVSD/
    things_imgs_train.csv
    things_imgs_test.csv
    test_idx_intrain.npy
  data/
    TVSD/
      Preprocessed_data/
      DNN_feature_maps/
        clip_h14/
        vae_latents/
    THINGS/
      Images/
  models/
```

If your layout differs, pass custom paths with CLI flags.

## Quick Start

Run from repository root (the folder containing `FuzzyAlign`).

1. Preprocess TVSD data

```bash
python FuzzyAlign/TVSD/preprocessing/preprocessing.py \
  --data_root ./data/TVSD/ \
  --save_path ./data/TVSD/Preprocessed_data/
```

2. Build test-index mapping

```bash
python FuzzyAlign/TVSD/preprocessing/find_test100.py \
  --train_csv ./TVSD/things_imgs_train.csv \
  --test_csv ./TVSD/things_imgs_test.csv \
  --save_path ./TVSD/test_idx_intrain.npy
```

3. Extract image features (CLIP)

```bash
python FuzzyAlign/TVSD/dnn_feature_extraction/feature_maps_clip_h14.py \
  --project_dir ./data/THINGS/ \
  --data_dir ./data/TVSD/ \
  --train_csv ./TVSD/things_imgs_train.csv \
  --test_csv ./TVSD/things_imgs_test.csv
```

4. Extract VAE latents

```bash
python FuzzyAlign/TVSD/dnn_feature_extraction/feature_maps_vae.py \
  --project_dir ./data/THINGS/Images/ \
  --save_dir ./data/TVSD/DNN_feature_maps/vae_latents/ \
  --train_csv ./TVSD/things_imgs_train.csv \
  --test_csv ./TVSD/things_imgs_test.csv
```

5. Train alignment backbone

```bash
python FuzzyAlign/fuzzyalign.py \
  --result_path ./results/ \
  --eeg_data_path ./data/TVSD/Preprocessed_data/ \
  --img_data_root ./data/TVSD/DNN_feature_maps/ \
  --dnn clip_h14
```

6. Train reconstruction modules

```bash
python FuzzyAlign/reconstruction/image_condition.py \
  --eeg_data_path ./data/TVSD/Preprocessed_data/ \
  --img_data_path ./data/TVSD/DNN_feature_maps/ \
  --models_root ./models \
  --dnn clip_h14

python FuzzyAlign/reconstruction/vae_alignment.py \
  --models_root ./models \
  --vae_latent_root ./data/TVSD/DNN_feature_maps/vae_latents \
  --test_idx_path ./TVSD/test_idx_intrain.npy \
  --test_index_path ./TVSD/things_imgs_test.csv
```

7. Generate images

```bash
python FuzzyAlign/reconstruction/image_generation_high.py \
  --models_root ./models \
  --generated_root ./generated_imgs_high_level \
  --test_index_path ./TVSD/things_imgs_test.csv

python FuzzyAlign/reconstruction/image_generation_low.py \
  --models_root ./models \
  --vae_latent_root ./data/TVSD/DNN_feature_maps/vae_latents \
  --generated_low_root ./generated_imgs_low_level \
  --generated_mix_root ./generated_imgs_low_and_high_level \
  --test_idx_path ./TVSD/test_idx_intrain.npy \
  --test_index_path ./TVSD/things_imgs_test.csv
```

8. Evaluate reconstructions

```bash
python FuzzyAlign/metrics/recon_metrics_low.py \
  --THINGS_dir ./data/THINGS/Images \
  --test_index_path ./TVSD/things_imgs_test.csv \
  --generated_root ./generated_imgs_low_and_high_level \
  --results_root ./results

python FuzzyAlign/metrics/recon_metrics_high.py \
  --THINGS_dir ./data/THINGS/Images \
  --test_index_path ./TVSD/things_imgs_test.csv \
  --generated_root ./generated_imgs_high_level \
  --results_root ./results
```

## Notes

- `reconstruction/model.py` is expected by reconstruction scripts and is included in this folder.
- GPU IDs are still hardcoded in some scripts (`cuda:0`, `cuda:1`, etc.). Adjust these if needed.
- For reproducibility, set seeds and pin dependency versions in your environment.
