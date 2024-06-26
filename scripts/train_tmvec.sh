#!/bin/bash

#SBATCH -N 1
#SBATCH --tmp=4000
#SBATCH --time=24:00:00
#SBATCH --job-name=tmvec
#SBATCH --output=/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/tmvec_train/tm_vec.out
#SBATCH --error=/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/tmvec_train/tm_vec.err
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:32g
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10g

module load gcc/11.4.0
module load cuda/12.1.1
module load cudnn/8.9.2.26
export CUDA_HOME=$CUDA_BASE

# copy data to TMPDIR
export DATADIR=/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/tmvec_data
cp $DATADIR/esm/train_embeddings.h5py $TMPDIR
cp $DATADIR/tm_pairs.tsv $TMPDIR

export METAG_RANDOM_SEED=123
export METAG_DMODEL=2560
export METAG_NLAYER=4
export METAG_NHEADS=2
export METAG_IN_DIM=1024
export METAG_WARMUP_STEPS=1000
export METAG_TRAIN_PROP=0.90
export METAG_VAL_PROP=0.05
export METAG_TEST_PROP=0.05

export METAG_LR=0.001
export METAG_BSIZE=24
export METAG_SESSION=/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/tmvec_train/transformer_lr${METAG_LR}_dmodel${METAG_DMODEL}_nlayer${METAG_NLAYER}_cosine_sigmoid_test
export METAG_RANDOM_SEED=$RANDOM
export EPOCHS=40
export METAG_DATA=$TMPDIR/train_embeddings.h5py
export METAG_PAIRS=$TMPDIR/tm_pairs.tsv
export NUM_WORKERS=4

export PATH="$PATH:/cluster/apps/sfos/bin:/cluster/home/vbezshapkin/miniconda3/bin"
source activate tmvec_slim

# # run python and check cuda devices
# echo $(which python)
# echo $(python -c "import torch; print(torch.cuda.device_count())")

python /nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/tm_vec/train.py \
    --nodes 1 \
    --gpus 1 \
    --session ${METAG_SESSION} \
    --hdf_file ${METAG_DATA} \
    --tm_pairs ${METAG_PAIRS} \
    --lr0 ${METAG_LR} \
    --max-epochs ${EPOCHS} \
    --batch-size ${METAG_BSIZE} \
    --d_model ${METAG_DMODEL} \
    --num_layers ${METAG_NLAYER} \
    --dim_feedforward ${METAG_IN_DIM} \
    --nhead ${METAG_NHEADS} \
    --warmup_steps ${METAG_WARMUP_STEPS} \
    --train-prop ${METAG_TRAIN_PROP} \
    --val-prop ${METAG_VAL_PROP} \
    --test-prop ${METAG_TEST_PROP} \
    --num-workers ${NUM_WORKERS}
