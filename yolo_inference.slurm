#!/bin/bash
#SBATCH --job-name="yolo7-inferens"
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

python3 detect.py --weights runs/train/yolov7_tiny_sheep/weights/best.pt --conf 0.25 --img-size 640 --source inference/images/sheep/
