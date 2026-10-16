python train.py --lr 0.002 --batch_size 16 --M 4 --dataset_name cub --alpha 12  --GPU 0,1 --bit 12 
python train.py --lr 0.002 --batch_size 16 --M 4 --dataset_name cub --alpha 12  --GPU 0,1 --bit 24 
python train.py --lr 0.002 --batch_size 16 --M 4 --dataset_name cub --alpha 12  --GPU 0,1 --bit 32 
python train.py --lr 0.002 --batch_size 16 --M 4 --dataset_name cub --alpha 12  --GPU 0,1 --bit 48 

python train.py --lr 0.002 --batch_size 16 --M 4 --dataset_name air --alpha 12  --GPU 0,1 --bit 12 
python train.py --lr 0.002 --batch_size 16 --M 4 --dataset_name air --alpha 12  --GPU 0,1 --bit 24 
python train.py --lr 0.002 --batch_size 16 --M 4 --dataset_name air --alpha 12  --GPU 0,1 --bit 32 
python train.py --lr 0.002 --batch_size 16 --M 4 --dataset_name air --alpha 12  --GPU 0,1 --bit 48 

python train.py --lr 0.002 --batch_size 64 --M 4 --dataset_name food101 --alpha 16 --GPU 0,1 --bit 12
python train.py --lr 0.002 --batch_size 64 --M 4 --dataset_name food101 --alpha 16 --GPU 0,1 --bit 24
python train.py --lr 0.002 --batch_size 64 --M 4 --dataset_name food101 --alpha 16 --GPU 0,1 --bit 32
python train.py --lr 0.002 --batch_size 64 --M 4 --dataset_name food101 --alpha 16 --GPU 0,1 --bit 48

python train.py --lr 0.002 --batch_size 16 --M 4 --dataset_name nabirds --alpha 12 --GPU 0,1 --bit 12
python train.py --lr 0.002 --batch_size 16 --M 4 --dataset_name nabirds --alpha 12 --GPU 0,1 --bit 24
python train.py --lr 0.002 --batch_size 16 --M 4 --dataset_name nabirds --alpha 22 --GPU 2,3 --bit 32
python train.py --lr 0.002 --batch_size 16 --M 4 --dataset_name nabirds --alpha 22 --GPU 4,5 --bit 48

python train.py --lr 0.002 --batch_size 64 --M 4 --dataset_name vegfru --alpha 16 --GPU 0,1 --bit 12
python train.py --lr 0.002 --batch_size 64 --M 4 --dataset_name vegfru --alpha 16 --GPU 0,1 --bit 24
python train.py --lr 0.002 --batch_size 64 --M 4 --dataset_name vegfru --alpha 16 --GPU 0,1 --bit 32
python train.py --lr 0.002 --batch_size 64 --M 4 --dataset_name vegfru --alpha 16 --GPU 0,1 --bit 48
