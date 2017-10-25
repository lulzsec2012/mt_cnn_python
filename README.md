
operating environment: tensorflow 1.0

run:
only use cpu:
python ./mtcnn_test.py

use GPU:
CUDA_VISIBLE_DEVICES=0 python ./mtcnn_test.py

--dataset_path : data path.  example :./jz_80val_0
