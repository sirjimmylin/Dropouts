#!/bin/bash
#SBATCH --job-name=bigearthnet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/users/ppate161/CACo/finetune/bigearthnet_%j.out
#SBATCH --error=/users/ppate161/CACo/finetune/bigearthnet_%j.err

module load anaconda3/2023.09-0-aqbc
source activate cacoenv

python /users/ppate161/CACo/finetune/train_bigearthnet.py
