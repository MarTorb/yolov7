#!/bin/bash
#SBATCH --job-name="prosjektoppgave-yolo7"
#SBATCH --account=ie-idi
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint=V100

module load Python/3.8.6-GCCcore-10.2.0
PyTorch/1.7.1-fosscuda-2020b-v100
module load CUDA/11.4.1

cd yolov7/

virtualenv venv

source venv/bin/activate

pip install -r requirements.txt

python train.py --epochs 300 --workers 8 --device 0 --batch-size 32 --data data/coco_sheep.yaml --img 412 412 --cfg cfg/training/yolov7-tiny.yaml --weights 'yolov7-tiny.pt' --name yolov7_tiny_sheep --hyp data/hyp.scratch.tiny.yaml

