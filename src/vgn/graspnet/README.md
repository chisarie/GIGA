# Installation

- Follow `/src/vgn/inference/README.md`

# Usage

```bash
export TRAIN_RAW=/home/chisari/datasets/centergrasp_g/giga_graspnet/train_raw
export TRAIN_PROCESSED=/home/chisari/datasets/centergrasp_g/giga_graspnet/train_processed/
python scripts/generate_data_parallel.py
python scripts/clean_balance_data.py TRAIN_RAW
python scripts/construct_dataset_parallel.py --num-proc 40 --single-view --add-noise dex TRAIN_RAW TRAIN_PROCESSED
python scripts/save_occ_data_parallel.py TRAIN_RAW 100000 2 --num-proc 40
python scripts/train_giga.py --dataset TRAIN_PROCESSED --dataset_raw TRAIN_RAW
```