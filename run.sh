# CUDA_VISIBLE_DEVICES=0 python cifar10_2ode.py --batch_size 1024
CUDA_VISIBLE_DEVICES=1 python cifar10_2ode.py --batch_size 1024 --use_gradloss --use_cutmix
