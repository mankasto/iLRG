# Experiment reference

This page preserves the original experiment recipes while keeping the root README concise. Commands write new outputs to `results/runs/` and `results/reconstructions/`.

## Label restoration examples

```bash
# MNIST / FCN-3
python main.py --exp_name label_restoration_mnist --dataset MNIST_GRAY --model dnn --num_tries 50 --num_images 24 --compare

# CIFAR-100 / LeNet-5
python main.py --exp_name label_restoration_cifar100 --dataset CIFAR100 --model lenet5 --num_tries 50 --num_images 24 --compare

# ImageNet / ResNet-50
python main.py --exp_name label_restoration_imagenet --dataset ImageNet --model resnet50 --num_tries 50 --num_images 24 --compare
```

## Batch-size and architecture studies

```bash
# MNIST batch sizes
for n in 4 8 16; do
  python main.py --exp_name batch_size_mnist --dataset MNIST_GRAY --model dnn --num_tries 50 --num_images "$n"
done

# FCN depth
for depth in 0 1 2; do
  python main.py --exp_name depth_mnist --dataset MNIST_GRAY --model dnn --num_tries 50 --num_images 64 --n_hidden "$depth"
done

# VGG variants
for model in vgg11 vgg13 vgg16 vgg19; do
  python main.py --exp_name architecture_cifar100 --dataset CIFAR100 --model "$model" --num_tries 50 --num_images 1024
done

# ResNet variants
for model in resnet18 resnet34 resnet50 resnet101 resnet152; do
  python main.py --exp_name architecture_imagenet --dataset ImageNet --model "$model" --num_tries 50 --num_images 2048
done
```

## Gradient inversion with iLRG

```bash
python main.py \
  --exp_name reconstruction_cifar100 \
  --seed 8888 \
  --dataset CIFAR100 \
  --model resnet18 \
  --num_tries 1 \
  --num_images 16 \
  --distribution random2 \
  --num_target_cls 10 \
  --rec_img \
  --restarts 1 \
  --signed \
  --boxed \
  --tv 8e-3 \
  --save_image \
  --max_iterations 240000 \
  --fix_labels
```

Add `--gt_labels` to reconstruct with ground-truth labels instead of labels restored by iLRG.

## Extreme label distributions

```bash
for n in 24 72 216 648; do
  python main.py --exp_name extreme_distribution --dataset CIFAR100 --model vgg16 --num_tries 20 --distribution custom_imbalanced --num_images "$n"
done
```

Append `--trained_model --epochs 100` to evaluate a trained model.

## Training-stage analysis

```bash
# MNIST iteration checkpoints
for iters in 100 200 300 400 500 600 700 800 900 1000; do
  python main.py --exp_name training_stage_mnist --dataset MNIST_GRAY --model dnn --num_tries 20 --num_images 8 --trained_model --iter_train --iters "$iters"
done

# CIFAR-100 epoch checkpoints
for epochs in 10 20 30 40 50 60 70 80 90 100; do
  python main.py --exp_name training_stage_cifar100 --dataset CIFAR100 --model vgg16 --num_tries 20 --num_images 64 --trained_model --epochs "$epochs"
done
```

## Error analysis and Soteria comparison

```bash
python main.py --exp_name error_analysis --dataset CIFAR100 --model vgg16 --num_tries 20 --num_images 64 --analysis
python main.py --exp_name error_analysis_epoch40 --dataset CIFAR100 --model vgg16 --num_tries 20 --num_images 64 --analysis --trained_model --epochs 40
python main.py --exp_name error_analysis_epoch100 --dataset CIFAR100 --model vgg16 --num_tries 20 --num_images 64 --analysis --trained_model --epochs 100
```

## Defense strategies

```bash
# Differential privacy
for std in 1e-4 1e-3 1e-2; do
  python main.py --exp_name defense_dp --dataset CIFAR100 --model resnet18 --num_tries 20 --num_images 24 --defense --defense_method dp --noise_std "$std"
done

# Gradient sparsification
for ratio in 10 20 40 80; do
  python main.py --exp_name defense_sparse --dataset CIFAR100 --model resnet18 --num_tries 20 --num_images 24 --defense --defense_method sparse --sparse_ratio "$ratio"
done
```

The archived outputs from the original release are documented in [`results/README.md`](../results/README.md).
