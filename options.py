"""Parser options."""

import argparse


def options():
    """Construct the central argument parser, filled with useful defaults."""
    parser = argparse.ArgumentParser(description='Instance-wise Batch Label Restoration and Image Reconstruction')

    # Basic settings
    parser.add_argument('--exp_name', default='Experiment', type=str)
    parser.add_argument('--cpu', action='store_true', help='Use cpu')
    parser.add_argument('--num_tries', default=50, type=int, help='Repetition times of an experiment')
    parser.add_argument('--num_images', default=24, type=int,
                        help='How many images should be recovered from the given gradient / Restoration batchsize')
    parser.add_argument('--seed', default=12, type=int, help='Random seed')
    parser.add_argument('--alpha', default=1, type=float, help='Factor for scaling outputs')
    parser.add_argument('--simplified', action='store_true',
                        help='Use simplified method, given class-wise label existences')
    parser.add_argument('--compare', action='store_true',
                        help='Compare our method with others')
    parser.add_argument('--estimate', action='store_true',
                        help='Use 1/n to estimate probabilities')
    parser.add_argument('--analysis', action='store_true',
                        help='Error analysis about four approximations and recovered embeddings & probs, etc')
    parser.add_argument('--ratio', default=0.00, type=float, help='Filter ratio to compute mean values')

    # Data settings
    parser.add_argument('--data_path', default='data', type=str)
    parser.add_argument('--dataset', default='CIFAR100', type=str)
    parser.add_argument('--split', default='train', type=str, help='Part of splitted dataset, default train')
    parser.add_argument('--distribution', default='random', type=str, help='Data distribution of a training batch,'
                                                                           'random, extreme, balanced, unique')
    parser.add_argument('--start_id', default=0, type=int,
                        help='The beginning image id for collecting data with extreme and balanced distribution')
    parser.add_argument('--num_classes', default=100, type=int)
    parser.add_argument('--num_uniform_cls', default=32, type=int,
                        help='Num of classes for collecting data with balanced distribution')
    parser.add_argument('--num_target_cls', default=5, type=int,
                        help='Num of classes for collecting data with random2 distribution')
    parser.add_argument('--max_size', default=32, type=int,
                        help='Max batch size for ImageNet')

    # Model settings
    parser.add_argument('--model', default='lenet5', type=str, help='model name.')
    parser.add_argument('--trained_model', action='store_true', help='Use a trained model.')
    ## Training settings
    parser.add_argument('--iter_train', action='store_true',
                        help='Train model with iterations setting instead of epochs')
    parser.add_argument('--iters', default=1000, type=int,
                        help='If using a trained model, how many iterations was it trained?')
    parser.add_argument('--epochs', default=10, type=int,
                        help='If using a trained model, how many epochs was it trained?')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batchsize for training, so is validation')
    parser.add_argument('--lr', default=0.1, type=float, help='Recommend 0.001 for adam series and 0.1 for sgd')
    parser.add_argument('--optimizer', default='SGD', type=str, help='AdamW, SGD, linear')
    parser.add_argument('--scheduler', default='linear', type=str, help='linear')
    parser.add_argument('--weight_decay', default=5e-4, help='Usually 5e-4')
    parser.add_argument('--warmup', action='store_true', help='Use warmup scheduler')
    parser.add_argument('--epoch_interval', default=10, type=int,
                        help='How many epochs to validate or save models')
    parser.add_argument('--iter_interval', default=100, type=int,
                        help='How many iterations to validate or save models')
    parser.add_argument('--mid_save', action='store_true', help='Save middle trained models')
    parser.add_argument('--model_path', default='models', type=str)
    parser.add_argument('--dryrun', action='store_true', help='Run everything for just one step to test functionality')
    ## End training settings
    parser.add_argument('--batchnorm', action='store_true', help='Use batchnorm for lenet5 model')
    parser.add_argument('--dropout', action='store_true', help='Use dropout for vgg16 model')
    parser.add_argument('--silu', action='store_true', help='Use silu activation, may occur negative values')
    parser.add_argument('--leaky_relu', action='store_true',
                        help='Use leaky relu activation, may occur negative values')
    parser.add_argument('--n_dim', default=300, type=int,
                        help='Dimension of embedding (the input of classification layer)')
    parser.add_argument('--n_hidden', default=1, type=int,
                        help='Num of hidden layers')

    # Defense settings
    parser.add_argument('--defense', action='store_true', help='Defense against the attack')
    parser.add_argument('--defense_method', default='dp', type=str,
                        help='dp(additive noise) or clip or sparse or perturb(soteria)')
    parser.add_argument('--noise_std', default=0.001, type=float)
    parser.add_argument('--clip_bound', default=4, type=int)
    parser.add_argument('--sparse_ratio', default=10, type=int)
    parser.add_argument('--prune_ratio', default=10, type=int)

    # Rec images settings
    parser.add_argument('--rec_img', action='store_true', help='Reconstruct images based on our attack, here IG')
    parser.add_argument('--fix_labels', action='store_true', help='Fix labels')
    parser.add_argument('--gt_labels', action='store_true', help='Fix labels with the gt')
    parser.add_argument('--optim', default='ig', type=str, help='IG or DLG')
    parser.add_argument('--restarts', default=1, type=int, help='How many restarts to run')
    parser.add_argument('--cost_fn', default='sim', type=str, help='Choice of cost function')
    parser.add_argument('--indices', default='def', type=str, help='Choice of indices from the parameter list')
    parser.add_argument('--weights', default='equal', type=str, help='Weigh the parameter list differently')
    parser.add_argument('--rec_lr', default=None, type=float, help='Learning rate for reconstruction')
    parser.add_argument('--rec_optimizer', default='adam', type=str, help='Optimizer for reconstruction')
    parser.add_argument('--signed', action='store_true', help='Use signed gradients, recommend true')
    parser.add_argument('--boxed', action='store_true', help='Use box constraints, recommend true')
    parser.add_argument('--scoring_choice', default='loss', type=str,
                        help='How to find the best image between all restarts')
    parser.add_argument('--init', default='randn', type=str, help='Choice of image initialization')
    parser.add_argument('--tv', default=1e-6, type=float, help='Weight of TV penalty')
    parser.add_argument('--l2', default=1e-6, type=float, help='Weight of l2 norm')
    parser.add_argument('--max_iterations', default=8000, type=int, help='Max iterations of reconstruction')
    parser.add_argument('--loss_thresh', default=1e-4, type=float, help='Loss threshold for early stopping')

    # Files and folders:
    parser.add_argument('--save_image', action='store_true', help='Save the output to a file.')
    # parser.add_argument('--image_dir', default='images/Experiment', type=str)
    return parser
