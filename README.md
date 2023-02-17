# Instance-wise Batch Label Restoration via Gradients In Federated Learning.<br>

The code for implementing our ***instance-wise Batch Label Restoration from Gradients* (*iLRG*)**, which is an analytic method to recover the number of instances per class via batch-averaged gradients in *Federated Learning* (*FL*). **The paper has been accepted for presentation at the ICLR2023 conference and can be found at <https://openreview.net/forum?id=FIrQfNSOoTr>**.

## Requirements

The project builds on the basic torch environment and several common libraries .

Here is a simple instruction to install the essential python libraries:

```bash
pip install -i requirements.txt
```

## Datasets and Models

We adopt three classical image classification datasets (i.e., MNIST, CIFAR100 and ImageNet) and four models (i.e., FCN-3, LeNet-5, VGG-series and ResNet-series, where FCN-3 is a 3-layer fully-connected network with a hidden layer dimension of 300).  The ImageNet dataset needs to be downloaded and placed in *./data* directory, and the rest of the datasets will be downloaded automatically.


## Running Experiments

### Basic Options

| Option/Parameter            | Help         |
| --------------------------- | ------------------:|
| exp_name                    | Experiment Name/Directory |
| seed                        | Random Seed        |
| num_tries                   | Number of Repetitions |
| num_images                  | Number of Images/Batch Size |
| dataset                     | Dataset Name      |
| split                       | 'train' or 'val' or 'all'|
| distribution                | Default Random    |
| num_classes                 | Number of Classes |
| model                       | Model Name     |
| n_hidden                    | Number of Hidden Layers for FCN |
| dropout                     | Use *dropout* for vgg16 model |
| batchnorm                   | Use *batchnorm* for lenet5 model |
| silu                        | Use *silu* activation for lenet5 model|
| trained_model               | Use a trained model  |
| epochs                      | Epochs of trained model |
| simplified                  | Use Simplified iLRG (Given classes) |
| estimate                    | Use 1/n to estimate probabilities |
| compare                     | Compare with Prior Works |
| analysis                    | Print MSEs and CosSims, etc|

### Experiment 1 (Examples)
```bash
python3 main.py --exp_name Experiment1 --dataset MNIST_GRAY --model dnn --num_tries 50 --num_images 24 --compare
```
```bash
python3 main.py --exp_name Experiment1 --dataset CIFAR100 --model lenet5 --num_tries 50 --num_images 24 --compare
python3 main.py --exp_name Experiment1 --dataset CIFAR100 --model lenet5 --num_tries 50 --num_images 24 --compare --silu
```
```bash
python3 main.py --exp_name Experiment1 --dataset ImageNet --model resnet50 --num_tries 50 --num_images 24 --compare
```

### Experiment 2 (Examples)
```bash
python3 main.py --exp_name Experiment2 --dataset MNIST_GRAY --model dnn --num_tries 50 --num_images 4
python3 main.py --exp_name Experiment2 --dataset MNIST_GRAY --model dnn --num_tries 50 --num_images 8
python3 main.py --exp_name Experiment2 --dataset MNIST_GRAY --model dnn --num_tries 50 --num_images 16
 
python3 main.py --exp_name Experiment2 --dataset MNIST_GRAY --model dnn --num_tries 50 --num_images 64 --n_hidden 0
python3 main.py --exp_name Experiment2 --dataset MNIST_GRAY --model dnn --num_tries 50 --num_images 64 --n_hidden 1
python3 main.py --exp_name Experiment2 --dataset MNIST_GRAY --model dnn --num_tries 50 --num_images 64 --n_hidden 2
```
```bash
python3 main.py --exp_name Experiment2 --dataset CIFAR100 --model vgg11 --num_tries 50 --num_images 1024 
python3 main.py --exp_name Experiment2 --dataset CIFAR100 --model vgg13 --num_tries 50 --num_images 1024 
python3 main.py --exp_name Experiment2 --dataset CIFAR100 --model vgg16 --num_tries 50 --num_images 1024 
python3 main.py --exp_name Experiment2 --dataset CIFAR100 --model vgg19 --num_tries 50 --num_images 1024 
```
```bash
python3 main.py --exp_name Experiment2 --dataset ImageNet --model resnet18 --num_tries 50 --num_images 2048 
python3 main.py --exp_name Experiment2 --dataset ImageNet --model resnet34 --num_tries 50 --num_images 2048 
python3 main.py --exp_name Experiment2 --dataset ImageNet --model resnet50 --num_tries 50 --num_images 2048 
python3 main.py --exp_name Experiment2 --dataset ImageNet --model resnet101 --num_tries 50 --num_images 2048 
python3 main.py --exp_name Experiment2 --dataset ImageNet --model resnet152 --num_tries 50 --num_images 2048 
```

### Experiment 3 Improved [IG](https://github.com/JonasGeiping/invertinggradients) (Examples)
####  Options

| Option/Parameter            | Help         |
| --------------------------- | ------------------:|
| rec_img                     | Reconstruct Images |
| optim                       | Optimization Method, Default IG |
| restarts                    | Number of Repetitions for Reconstruction |
| cost_fn                     | Cost Function |
| rec_lr                      | Learning Rate for Reconstruction |
| rec_optimizer               | Optimizer for Reconstruction |
| signed                      | Use Signed Gradients |
| boxed                       | Use Boxed Constraints |
| init                        | Image Initialization     |
| tv                          | Weight of TV Penalty |
| l2                          | Weight of L2 Normalization |
| max_iterations              | Maximum Iterations of Reconstuction |
| fix_labels                  | Fix Labels |
| gt_labels                   | Fix Labels with the Ground-truth, otherwise with our iLRG |
| save_images                 | Save Recovered and GT Images |

```bash
python3 main.py --exp_name Experiment3 --seed 8888 --dataset CIFAR100 --model resnet18 --num_tries 1 --num_images 16 --distribution random2 --num_target_cls 10 --rec_img --restarts 1 --signed --boxed --tv 8e-3 --save_image --max_iterations 240000 
python3 main.py --exp_name Experiment3 --seed 8888 --dataset CIFAR100 --model resnet18 --num_tries 1 --num_images 16 --distribution random2 --num_target_cls 10 --rec_img --restarts 1 --signed --boxed --tv 8e-3 --save_image --max_iterations 240000 --fix_labels
```
### Experiments for Extreme Distribution (Examples)

```bash
python3 main.py --exp_name extreme_distribution --dataset CIFAR100 --model vgg16 --num_tries 20 --distribution custom_imbalanced --num_images 24
python3 main.py --exp_name extreme_distribution --dataset CIFAR100 --model vgg16 --num_tries 20 --distribution custom_imbalanced --num_images 72
python3 main.py --exp_name extreme_distribution --dataset CIFAR100 --model vgg16 --num_tries 20 --distribution custom_imbalanced --num_images 216
python3 main.py --exp_name extreme_distribution --dataset CIFAR100 --model vgg16 --num_tries 20 --distribution custom_imbalanced --num_images 648
```
```bash
python3 main.py --exp_name extreme_distribution --dataset CIFAR100 --model vgg16 --num_tries 20 --distribution custom_imbalanced --num_images 24 --trained_model --epochs 100
python3 main.py --exp_name extreme_distribution --dataset CIFAR100 --model vgg16 --num_tries 20 --distribution custom_imbalanced --num_images 72 --trained_model --epochs 100
python3 main.py --exp_name extreme_distribution --dataset CIFAR100 --model vgg16 --num_tries 20 --distribution custom_imbalanced --num_images 216 --trained_model --epochs 100
python3 main.py --exp_name extreme_distribution --dataset CIFAR100 --model vgg16 --num_tries 20 --distribution custom_imbalanced --num_images 648 --trained_model --epochs 100
```

### Experiments for Training Stages (Examples)

```bash
python3 main.py --exp_name train_stage --dataset MNIST_GRAY --model dnn --num_tries 20 --num_images 8
python3 main.py --exp_name train_stage --dataset MNIST_GRAY --model dnn --num_tries 20 --num_images 8 --trained_model --iter_train --iters 100
python3 main.py --exp_name train_stage --dataset MNIST_GRAY --model dnn --num_tries 20 --num_images 8 --trained_model --iter_train --iters 200
python3 main.py --exp_name train_stage --dataset MNIST_GRAY --model dnn --num_tries 20 --num_images 8 --trained_model --iter_train --iters 300
python3 main.py --exp_name train_stage --dataset MNIST_GRAY --model dnn --num_tries 20 --num_images 8 --trained_model --iter_train --iters 400
python3 main.py --exp_name train_stage --dataset MNIST_GRAY --model dnn --num_tries 20 --num_images 8 --trained_model --iter_train --iters 500
python3 main.py --exp_name train_stage --dataset MNIST_GRAY --model dnn --num_tries 20 --num_images 8 --trained_model --iter_train --iters 600
python3 main.py --exp_name train_stage --dataset MNIST_GRAY --model dnn --num_tries 20 --num_images 8 --trained_model --iter_train --iters 700
python3 main.py --exp_name train_stage --dataset MNIST_GRAY --model dnn --num_tries 20 --num_images 8 --trained_model --iter_train --iters 800
python3 main.py --exp_name train_stage --dataset MNIST_GRAY --model dnn --num_tries 20 --num_images 8 --trained_model --iter_train --iters 900
python3 main.py --exp_name train_stage --dataset MNIST_GRAY --model dnn --num_tries 20 --num_images 8 --trained_model --iter_train --iters 1000
```
```bash
python3 main.py --exp_name train_stage --dataset CIFAR100 --model vgg16 --num_tries 20 --num_images 64
python3 main.py --exp_name train_stage --dataset CIFAR100 --model vgg16 --num_tries 20 --num_images 64 --trained_model --epochs 10
python3 main.py --exp_name train_stage --dataset CIFAR100 --model vgg16 --num_tries 20 --num_images 64 --trained_model --epochs 20
python3 main.py --exp_name train_stage --dataset CIFAR100 --model vgg16 --num_tries 20 --num_images 64 --trained_model --epochs 30
python3 main.py --exp_name train_stage --dataset CIFAR100 --model vgg16 --num_tries 20 --num_images 64 --trained_model --epochs 40
python3 main.py --exp_name train_stage --dataset CIFAR100 --model vgg16 --num_tries 20 --num_images 64 --trained_model --epochs 50
python3 main.py --exp_name train_stage --dataset CIFAR100 --model vgg16 --num_tries 20 --num_images 64 --trained_model --epochs 60
python3 main.py --exp_name train_stage --dataset CIFAR100 --model vgg16 --num_tries 20 --num_images 64 --trained_model --epochs 70
python3 main.py --exp_name train_stage --dataset CIFAR100 --model vgg16 --num_tries 20 --num_images 64 --trained_model --epochs 80
python3 main.py --exp_name train_stage --dataset CIFAR100 --model vgg16 --num_tries 20 --num_images 64 --trained_model --epochs 90
python3 main.py --exp_name train_stage --dataset CIFAR100 --model vgg16 --num_tries 20 --num_images 64 --trained_model --epochs 100
```

### Experiments for ErrorAnalysis and  Comparison with Soteria (Examples)

```bash
python3 main.py --exp_name analysis --dataset CIFAR100 --model vgg16 --num_tries 20 --num_images 64 --analysis
python3 main.py --exp_name analysis --dataset CIFAR100 --model vgg16 --num_tries 20 --num_images 64 --analysis --trained_model --epochs 100
python3 main.py --exp_name analysis --dataset CIFAR100 --model vgg16 --num_tries 20 --num_images 64 --analysis --trained_model --epochs 40
```

### Experiments for DefenseStrategies (Examples)

####  Options

| Option/Parameter            | Help         |
| --------------------------- | ------------------:|
| defense                     | Use Defense Strategies against Attacks |
| defense_method              | Defense Strategies, Including DP, Clipping, Sparsification and Soteria Pruning |
| noise_std                   | The STD of Gaussian Noise for DP, Default 0.001|
| clip_bound                  | The Clipping Bound for Gradient Clipping, Default 4 |
| sparse_ratio                | The Sparsification Ratio for Gradient Sparsification, Default 10%  |
| prune_ratio                 | The Pruning Ratio for Soteria, Default 10% (Only Applied to ResNet18) |

####  DP (Differential Privacy)
```bash
python3 main.py --exp_name defense --dataset CIFAR100 --model resnet18 --num_tries 20 --num_images 24
python3 main.py --exp_name defense --dataset CIFAR100 --model resnet18 --num_tries 20 --num_images 24 --defense --defense_method dp --noise_std 1e-4
python3 main.py --exp_name defense --dataset CIFAR100 --model resnet18 --num_tries 20 --num_images 24 --defense --defense_method dp --noise_std 1e-3
python3 main.py --exp_name defense --dataset CIFAR100 --model resnet18 --num_tries 20 --num_images 24 --defense --defense_method dp --noise_std 1e-2
```
```bash
python3 main.py --exp_name defense --dataset CIFAR100 --model resnet18 --num_tries 20 --num_images 24 --trained_model --epochs 100
python3 main.py --exp_name defense --dataset CIFAR100 --model resnet18 --num_tries 20 --num_images 24 --defense --defense_method dp --noise_std 1e-4 --trained_model --epochs 100 
python3 main.py --exp_name defense --dataset CIFAR100 --model resnet18 --num_tries 20 --num_images 24 --defense --defense_method dp --noise_std 1e-3 --trained_model --epochs 100
python3 main.py --exp_name defense --dataset CIFAR100 --model resnet18 --num_tries 20 --num_images 24 --defense --defense_method dp --noise_std 1e-2 --trained_model --epochs 100
```

####  GS (Gradient Sparsification)

```bash
python3 main.py --exp_name defense --dataset CIFAR100 --model resnet18 --num_tries 20 --num_images 24 --defense --defense_method sparse --sparse_ratio 10
python3 main.py --exp_name defense --dataset CIFAR100 --model resnet18 --num_tries 20 --num_images 24 --defense --defense_method sparse --sparse_ratio 20
python3 main.py --exp_name defense --dataset CIFAR100 --model resnet18 --num_tries 20 --num_images 24 --defense --defense_method sparse --sparse_ratio 40
python3 main.py --exp_name defense --dataset CIFAR100 --model resnet18 --num_tries 20 --num_images 24 --defense --defense_method sparse --sparse_ratio 80
```
```bash
python3 main.py --exp_name defense --dataset CIFAR100 --model resnet18 --num_tries 20 --num_images 24 --defense --defense_method sparse --sparse_ratio 10 --trained_model --epochs 100 
python3 main.py --exp_name defense --dataset CIFAR100 --model resnet18 --num_tries 20 --num_images 24 --defense --defense_method sparse --sparse_ratio 20 --trained_model --epochs 100
python3 main.py --exp_name defense --dataset CIFAR100 --model resnet18 --num_tries 20 --num_images 24 --defense --defense_method sparse --sparse_ratio 40 --trained_model --epochs 100
python3 main.py --exp_name defense --dataset CIFAR100 --model resnet18 --num_tries 20 --num_images 24 --defense --defense_method sparse --sparse_ratio 80 --trained_model --epochs 100
```

## Results

We have placed our experimental logs in *./logs* directory including the main text and appendix. 

### Comparison with Prior Works
| Model         | Dataset  | LeAcc | LnAcc | CosSim (Prob) | iDLG LeAcc | GI LeAcc | RLG LeAcc|
| :---------------| :------- | ----- | :---- | :----  | :---- | :---- | :---- |
| FCN-3         | MNIST    | **1.000** | **0.994** | **0.979**  | 0.514 | 1.000 | 0.932 |
| LeNet-5       | CIFAR100 | **1.000** | **1.000** | **1.000**  | 1.000 | 1.000 | 1.000 |
| LeNet-S *     | CIFAR100 | **1.000** | **1.000** | **1.000**  | 0.150 | 0.213 | 1.000 |
| VGG-16        | ImageNet | **1.000** | **1.000** | **1.000**  | 1.000 | 1.000 | 0.981 |
| ResNet-50    | ImageNet | **1.000** | **1.000** | **1.000**  | 1.000 | 1.000 | 1.000 |

####   Improved Gradient Inversion Attack With Our iLRG

<img src="README_md_files/f67ab870-3e10-11ed-9ab9-579d6cfc3a57.jpeg?v=1&type=image" alt="image" style="zoom: 67%;" />

Batch image reconstruction on MNIST (FCN-3, BS50) and CIFAR100 (ResNet-18, BS16) compared with IG. We assign a specific label to each instance after label restoration at 100% accuracy. The 6 best visual images are selected to display and calculate the metrics.

## Citation

```
@inproceedings{
    ma2023instancewise,
    title={Instance-wise Batch Label Restoration via Gradients in Federated Learning},
    author={Kailang Ma and Yu Sun and Jian Cui and Dawei Li and Zhenyu Guan and Jianwei Liu},
    booktitle={The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=FIrQfNSOoTr}
}
```

## License

This project is released under the MIT License.
