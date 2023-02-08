import time
import consts
import rec_imgs
import torchvision
from utils import *
import torch.nn as nn
from defense import params
from options import options
from methods import get_emb
from models import get_model
from datasets import get_dataset
from train import Classification, train, iter_train

if __name__ == '__main__':
    args = options().parse_args()
    args.log_dir = os.path.join('logs', args.exp_name)
    args.image_dir = os.path.join('images', args.exp_name)

    # Defend params
    noise_std = args.noise_std
    clip_bound = args.clip_bound
    sparse_ratio = args.sparse_ratio
    prune_ratio = args.prune_ratio

    # Set random seeds
    set_random_seed(args.seed)
    # Get log file handle
    log_file = get_log_file(args.log_dir)
    # Log current time
    current_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
    print_util('running experiments at {}'.format(current_time), log_file=log_file)
    print_util(get_params_string(vars(args)), log_file=log_file)

    # Get dataset
    dataset_params = get_dataset(dataset=args.dataset,
                                 data_path=args.data_path,
                                 model=args.model,
                                 aug=False, )
    img_shape, num_classes, channel, hidden, dataset = dataset_params
    train_data, valid_data = dataset
    target_data = train_data if args.split == 'train' else valid_data
    args.num_classes = num_classes
    device = get_device(use_cuda=False if args.cpu else True)
    setup = dict(device=device, dtype=torch.float)

    # Get model
    model = get_model(model_name=args.model,
                      net_params=(num_classes, channel, hidden),
                      device=device,
                      n_hidden=args.n_hidden,
                      n_dim=args.n_dim,
                      batchnorm=args.batchnorm,
                      dropout=args.dropout,
                      silu=args.silu,
                      leaky_relu=args.leaky_relu)
    model = model.to(device)

    # Load a trained model
    if args.trained_model:
        file = f'{args.model}_{args.dataset}_Iter{args.iters}.pth' if args.iter_train else f'{args.model}_{args.dataset}_Epoch{args.epochs}.pth'
        try:
            model.load_state_dict(torch.load(os.path.join(args.model_path, file), map_location=device))
            print_util(f'Model loaded from file {file}.', log_file=log_file)
        except FileNotFoundError:
            print_util('Training the model ...', log_file=log_file)
            # Training configs
            defs = Config({'iter_train': args.iter_train,
                           'epochs': args.epochs,
                           'iterations': args.iters,
                           'batch_size': args.batch_size,
                           'optimizer': args.optimizer,
                           'lr': args.lr,
                           'scheduler': args.scheduler,
                           'weight_decay': args.weight_decay,
                           'warmup': args.warmup,
                           'epoch_interval': args.epoch_interval,
                           'iter_interval': args.iter_interval,
                           'dryrun': args.dryrun,
                           'model': args.model,
                           'dataset': args.dataset,
                           'mid_save': args.mid_save,
                           'save_dir': args.model_path})
            # Get dataloader
            train_loader = get_dataloader(train_data, batch_size=defs.batch_size, shuffle=True)
            valid_loader = get_dataloader(valid_data, batch_size=defs.batch_size, shuffle=False)
            if args.iter_train:
                stats = iter_train(model, Classification(), train_loader, valid_loader, defs, setup=setup)
            else:
                stats = train(model, Classification(), train_loader, valid_loader, defs, setup=setup)

    if args.rec_img:
        dataset_lower = 'mnist' if args.dataset.lower() == 'mnist_gray' else args.dataset.lower()
        dm = torch.as_tensor(getattr(consts, f'{dataset_lower}_mean'))[:, None, None].to(device)
        ds = torch.as_tensor(getattr(consts, f'{dataset_lower}_std'))[:, None, None].to(device)
        if args.optim == 'ig':
            rec_config = dict(signed=args.signed,
                              boxed=True,
                              cost_fn=args.cost_fn,
                              indices=args.indices,
                              weights=args.weights,
                              lr=args.rec_lr if args.rec_lr is not None else 0.1,
                              optim='adam',
                              restarts=args.restarts,
                              max_iterations=args.max_iterations,
                              total_variation=args.tv,
                              l2_norm=args.l2,
                              init=args.init,
                              filter='median' if args.dataset == 'ImageNet' else 'none',
                              lr_decay=True,
                              scoring_choice=args.scoring_choice)
        elif args.optim == 'dlg':
            rec_config = dict(signed=False,
                              boxed=False,
                              cost_fn='l2',
                              indices='def',
                              weights='equal',
                              lr=args.rec_lr if args.rec_lr is not None else 1.0,
                              optim='LBFGS',
                              restarts=args.restarts,
                              max_iterations=args.max_iterations,  # 500
                              total_variation=args.tv,
                              l2_norm=args.l2,
                              init=args.init,
                              filter='none',
                              lr_decay=False,
                              scoring_choice=args.scoring_choice)

    model.eval()
    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Multiple experiments
    iLRG_leaccs, iLRG_lnaccs, iLRG_irecs = AverageMeter(), AverageMeter(), AverageMeter()
    iDLG_leaccs, GI_leaccs, SVD_leaccs = AverageMeter(), AverageMeter(), AverageMeter()

    mtp_approx1_mses, mtp_approx2_mses, mtp_approx3_mses, mtp_approx4_mses = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    mtp_scale_mses, mtp_approx_out_mses, mtp_approx_prob_mses = AverageMeter(), AverageMeter(), AverageMeter()
    mtp_emb_mses, mtp_out_mses, mtp_prob_mses = AverageMeter(), AverageMeter(), AverageMeter()

    mtp_approx1_mres, mtp_approx2_mres, mtp_approx3_mres, mtp_approx4_mres = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    mtp_scale_mres, mtp_approx_out_mres, mtp_approx_prob_mres = AverageMeter(), AverageMeter(), AverageMeter()
    mtp_emb_mres, mtp_out_mres, mtp_prob_mres = AverageMeter(), AverageMeter(), AverageMeter()

    mtp_emb_sims, mtp_out_sims, mtp_prob_sims = AverageMeter(), AverageMeter(), AverageMeter()
    mtp_soteria_mses, mtp_soteria_sims = AverageMeter(), AverageMeter()

    if 'custom' in args.distribution:
        mtp_0_cnt, mtp_18_cnt, mtp_92_cnt = AverageMeter(), AverageMeter(), AverageMeter()

    target_id = args.start_id
    max_thresh = 10

    for exp_id in range(args.num_tries):
        print_util('Start Experiment {}'.format(exp_id + 1), log_file=log_file)
        # Load batch data
        target_id = target_id % len(target_data)
        print_util('start_id: ' + str(target_id), log_file=log_file)
        gt_data, gt_label, target_id = get_data(dataset=target_data,
                                                num_images=args.num_images,
                                                num_classes=args.num_classes,
                                                data_distribution=args.distribution,
                                                start_id=target_id,
                                                num_uniform_cls=args.num_uniform_cls,
                                                num_target_cls=args.num_target_cls,
                                                device=device)
        print('Finish getting data!')
        # Forward Process
        if args.dataset == 'ImageNet' and args.num_images > args.max_size:
            assert args.num_images % args.max_size == 0, 'Number of images should be an integral multiple of max size'
            split_data = torch.split(gt_data, args.max_size, 0)
            split_label = torch.split(gt_label, args.max_size, 0)
            num_batch = args.num_images // args.max_size
            grads, embeddings, outs = [], [], []
            for i in range(num_batch):
                split_outs, split_embeddings = model(split_data[i])
                split_grads = get_grads(outs=split_outs, labels=split_label[i], model=model, loss_fn=criterion,
                                        rec=args.rec_img)
                grads.append(split_grads)
                outs.append(split_outs)
                embeddings.append(split_embeddings)
                print('Finish splitting {}!'.format(i))
            grads = np.stack(np.array(grads))
            grads = np.mean(grads, axis=0)
            embeddings = np.stack(np.array(embeddings))
            outs = np.stack(np.array(outs))
            # Collect several class-wise data for error analysis
            if args.analysis:
                indexes, cls_embeddings, cls_outs, cls_probs, cls_wgrad, cls_bgrad = collect_cls_variables(
                    embeddings=embeddings,
                    outs=outs,
                    gt_label=gt_label,
                    num_classes=args.num_classes,
                    model=model,
                    loss_fn=criterion)
        else:
            outs, embeddings = model(gt_data)
            # Collect several class-wise data for error analysis
            if args.analysis:
                indexes, cls_embeddings, cls_outs, cls_probs, cls_wgrad, cls_bgrad = collect_cls_variables(
                    embeddings=embeddings,
                    outs=outs,
                    gt_label=gt_label,
                    num_classes=args.num_classes,
                    model=model,
                    loss_fn=criterion)
            # Get Batch-averaged gradients
            grads = get_grads(outs=outs, labels=gt_label, model=model, loss_fn=criterion, rec=args.rec_img)
            print('Finish getting grads!')
        print('Finish forward process!')
        probs = torch.softmax(outs, dim=-1)
        preds = torch.max(probs, 1)[1].cpu()
        correct = (preds == gt_label.cpu()).sum().item()
        print_util('%d/%d, Acc for this batch: %.3f' % (correct, args.num_images, correct / args.num_images),
                   log_file=log_file)

        w_grad, b_grad = grads[-2], grads[-1]
        # Lossy transformation to gradients to defend leakage
        if args.defense:
            w_grad, b_grad = degrade_grads(grads=[w_grad, b_grad],
                                           defense_method=args.defense_method,
                                           param_value=eval(params[args.defense_method]),
                                           model_name=args.model,
                                           images=gt_data,
                                           model=model)
        cls_rec_probs = []
        if args.analysis:
            approx1_mses, approx2_mses, approx3_mses, approx4_mses = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            scale_mses, approx_out_mses, approx_prob_mses = AverageMeter(), AverageMeter(), AverageMeter()
            emb_mses, out_mses, prob_mses = AverageMeter(), AverageMeter(), AverageMeter()

            approx1_mres, approx2_mres, approx3_mres, approx4_mres = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            scale_mres, approx_out_mres, approx_prob_mres = AverageMeter(), AverageMeter(), AverageMeter()
            emb_mres, out_mres, prob_mres = AverageMeter(), AverageMeter(), AverageMeter()

            emb_sims, out_sims, prob_sims = AverageMeter(), AverageMeter(), AverageMeter()
            soteria_mses, soteria_sims = AverageMeter(), AverageMeter()

        for i in range(num_classes):
            # Recover class-specific embeddings and probabilities
            cls_rec_emb = get_emb(w_grad[i], b_grad[i])
            # if (not args.silu) and (not args.leaky_relu):
            #     cls_rec_emb = torch.where(cls_rec_emb < 0., torch.full_like(cls_rec_emb, 0), cls_rec_emb)
            # cls_rec_emb = torch.where(w_grad[i] < 0., torch.full_like(w_grad[i], 0), w_grad[i])
            cls_rec_prob = post_process_emb(embedding=cls_rec_emb,
                                            model=model,
                                            device=device,
                                            alpha=args.alpha)
            cls_rec_probs.append(cls_rec_prob)

            # Calculate MSEs.
            if args.analysis:
                if len(indexes[i]) > 0:
                    avg_opt1 = model.fc(cls_embeddings[i])
                    avg_sfm_opt1 = torch.softmax(avg_opt1, dim=0)
                    avg_sfm_opt2 = torch.softmax(cls_outs[i], dim=0)
                    avg_rec_opt = model.fc(cls_rec_emb.to(device))
                    # print(cls_rec_emb.max(), cls_embeddings[i].max(), w_grad[i].max())
                    emb_mse = torch.mean((cls_rec_emb.cpu() - cls_embeddings[i].cpu()) ** 2)
                    emb_mre = torch.abs((cls_rec_emb.cpu() - cls_embeddings[i].cpu()) / cls_embeddings[i].cpu())
                    emb_mre = torch.where(torch.isinf(emb_mre), torch.full_like(emb_mre, 0), emb_mre)
                    emb_mre = torch.mean(torch.where(torch.isnan(emb_mre), torch.full_like(emb_mre, 0), emb_mre))
                    emb_sim = torch.cosine_similarity(cls_rec_emb.cpu(), cls_embeddings[i].cpu(), dim=-1)

                    out_mse = torch.mean((avg_rec_opt.cpu() - cls_outs[i].cpu()) ** 2)
                    out_mre = torch.abs((avg_rec_opt.cpu() - cls_outs[i].cpu()) / cls_outs[i].cpu())
                    out_mre = torch.where(torch.isinf(out_mre), torch.full_like(out_mre, 0), out_mre)
                    out_mre = torch.mean(torch.where(torch.isnan(out_mre), torch.full_like(out_mre, 0), out_mre))
                    out_sim = torch.cosine_similarity(avg_rec_opt.cpu(), cls_outs[i].cpu(), dim=-1)

                    prob_mse = torch.mean((cls_rec_prob.cpu() - cls_probs[i].cpu()) ** 2)
                    prob_mre = torch.abs((cls_rec_prob.cpu() - cls_probs[i].cpu()) / cls_probs[i].cpu())
                    prob_mre = torch.where(torch.isinf(prob_mre), torch.full_like(prob_mre, 0), prob_mre)
                    prob_mre = torch.mean(torch.where(torch.isnan(prob_mre), torch.full_like(prob_mre, 0), prob_mre))
                    prob_sim = torch.cosine_similarity(cls_rec_prob.cpu(), cls_probs[i].cpu(), dim=-1)

                    soteria_mse = torch.mean((w_grad[i].cpu() - cls_embeddings[i].cpu()) ** 2)
                    soteria_sim = torch.cosine_similarity(w_grad[i].cpu(), cls_embeddings[i].cpu(), dim=-1)

                    approx1_mse = torch.mean((cls_wgrad[i].cpu() - cls_bgrad[i].cpu() * cls_embeddings[i].cpu()) ** 2)
                    approx1_mre = torch.abs(
                        (cls_wgrad[i].cpu() - cls_bgrad[i].cpu() * cls_embeddings[i].cpu()) / cls_wgrad[i].cpu())
                    approx1_mre = torch.where(torch.isinf(approx1_mre), torch.full_like(approx1_mre, 0), approx1_mre)
                    approx1_mre = torch.mean(
                        torch.where(torch.isnan(approx1_mre), torch.full_like(approx1_mre, 0), approx1_mre))

                    approx2_mse = (w_grad[i].cpu() - len(indexes[i]) / args.num_images * cls_wgrad[i].cpu()) ** 2
                    approx2_mse = torch.where(torch.isinf(approx2_mse), torch.full_like(approx2_mse, 0), approx2_mse)
                    approx2_mse = torch.mean(
                        torch.where(torch.isnan(approx2_mse), torch.full_like(approx2_mse, 0), approx2_mse))
                    approx2_mre = torch.abs(
                        (w_grad[i].cpu() - len(indexes[i]) / args.num_images * cls_wgrad[i].cpu()) / w_grad[i].cpu())
                    approx2_mre = torch.where(torch.isinf(approx2_mre), torch.full_like(approx2_mre, 0), approx2_mre)
                    approx2_mre = torch.mean(
                        torch.where(torch.isnan(approx2_mre), torch.full_like(approx2_mre, 0), approx2_mre))

                    approx3_mse = (b_grad[i].cpu() - len(indexes[i]) / args.num_images * cls_bgrad[i].cpu()) ** 2
                    # if i in [0, 18, 92]:
                    #     print(b_grad[i], cls_bgrad[i], len(indexes[i]) / args.num_images * cls_bgrad[i])
                    approx3_mse = torch.where(torch.isinf(approx3_mse), torch.full_like(approx3_mse, 0), approx3_mse)
                    approx3_mse = torch.mean(
                        torch.where(torch.isnan(approx3_mse), torch.full_like(approx3_mse, 0), approx3_mse))
                    approx3_mre = torch.abs(
                        (b_grad[i].cpu() - len(indexes[i]) / args.num_images * cls_bgrad[i].cpu()) / b_grad[i].cpu())
                    approx3_mre = torch.where(torch.isinf(approx3_mre), torch.full_like(approx3_mre, 0), approx3_mre)
                    approx3_mre = torch.mean(
                        torch.where(torch.isnan(approx3_mre), torch.full_like(approx3_mre, 0), approx3_mre))

                    approx4_mse = torch.mean((avg_sfm_opt1.cpu() - cls_probs[i].cpu()) ** 2)
                    approx4_mre = torch.abs((avg_sfm_opt1.cpu() - cls_probs[i].cpu()) / cls_probs[i].cpu())
                    approx4_mre = torch.where(torch.isinf(approx4_mre), torch.full_like(approx4_mre, 0), approx4_mre)
                    approx4_mre = torch.mean(
                        torch.where(torch.isnan(approx4_mre), torch.full_like(approx4_mre, 0), approx4_mre))

                    scale_mse = (w_grad[i].cpu() / b_grad[i].cpu() - cls_wgrad[i].cpu() / cls_bgrad[i].cpu()) ** 2
                    scale_mse = torch.where(torch.isinf(scale_mse), torch.full_like(scale_mse, 0), scale_mse)
                    scale_mse = torch.mean(
                        torch.where(torch.isnan(scale_mse), torch.full_like(scale_mse, 0), scale_mse))
                    scale_mre = torch.abs(
                        (w_grad[i].cpu() / b_grad[i].cpu() - cls_wgrad[i].cpu() / cls_bgrad[i].cpu()) / (
                                cls_wgrad[i] / cls_bgrad[i].cpu()))
                    scale_mre = torch.where(torch.isinf(scale_mre), torch.full_like(scale_mre, 0), scale_mre)
                    scale_mre = torch.mean(
                        torch.where(torch.isnan(scale_mre), torch.full_like(scale_mre, 0), scale_mre))

                    approx_out_mse = torch.mean((avg_opt1.cpu() - cls_outs[i].cpu()) ** 2)
                    approx_out_mre = torch.abs((avg_opt1.cpu() - cls_outs[i].cpu()) / cls_outs[i].cpu())
                    approx_out_mre = torch.where(torch.isinf(approx_out_mre), torch.full_like(approx_out_mre, 0),
                                                 approx_out_mre)
                    approx_out_mre = torch.mean(
                        torch.where(torch.isnan(approx_out_mre), torch.full_like(approx_out_mre, 0), approx_out_mre))

                    approx_prob_mse = torch.mean((avg_sfm_opt2.cpu() - cls_probs[i].cpu()) ** 2)
                    approx_prob_mre = torch.abs((avg_sfm_opt2.cpu() - cls_probs[i].cpu()) / cls_probs[i].cpu())
                    approx_prob_mre = torch.where(torch.isinf(approx_prob_mre), torch.full_like(approx_prob_mre, 0),
                                                  approx_prob_mre)
                    approx_prob_mre = torch.mean(
                        torch.where(torch.isnan(approx_prob_mre), torch.full_like(approx_prob_mre, 0), approx_prob_mre))

                    approx1_mses.update(approx1_mse.item())
                    approx2_mses.update(approx2_mse.item())
                    approx3_mses.update(approx3_mse.item())
                    approx4_mses.update(approx4_mse.item())

                    scale_mses.update(scale_mse.item())
                    approx_out_mses.update(approx_out_mse.item())
                    approx_prob_mses.update(approx_prob_mse.item())

                    emb_mses.update(emb_mse.item() if emb_mse.item() < max_thresh else max_thresh)
                    out_mses.update(out_mse.item())
                    prob_mses.update(prob_mse.item())

                    emb_sims.update(emb_sim.item())
                    out_sims.update(out_sim.item())
                    prob_sims.update(prob_sim.item())

                    approx1_mres.update(approx1_mre.item())
                    approx2_mres.update(approx2_mre.item())
                    approx3_mres.update(approx3_mre.item())
                    approx4_mres.update(approx4_mre.item())

                    scale_mres.update(scale_mre.item())
                    approx_out_mres.update(approx_out_mre.item())
                    approx_prob_mres.update(approx_prob_mre.item())

                    emb_mres.update(emb_mre.item())
                    out_mres.update(out_mre.item())
                    prob_mres.update(prob_mre.item())

                    soteria_sims.update(soteria_sim.item())
                    soteria_mses.update(soteria_mse.item())

                    print_util(
                        'Class %d | Approx1 MSE: %e | Approx2 MSE: %e | Approx3 MSE: %e | Approx4 MSE: %e'
                        % (i, approx1_mse.item(), approx2_mse.item(), approx3_mse.item(), approx4_mse.item()),
                        log_file=log_file)
                    print_util(
                        'Class %d | Scale MSE: %e | Approx Out MSE: %e | Approx Probabilities MSE: %e'
                        % (i, scale_mse.item(), approx_out_mse.item(), approx_prob_mse.item()),
                        log_file=log_file)
                    print_util(
                        'Class %d | Embedding MSE: %e | Out MSE: %e | Probabilities MSE: %e'
                        % (i, emb_mse.item(), out_mse.item(), prob_mse.item()),
                        log_file=log_file)
                    print_util(
                        'Class %d | Embedding Cosine Similarity: %.3f | Out Cosine Similarity: %.3f | Probabilities Cosine Similarity: %.3f'
                        % (i, emb_sim.item(), out_sim.item(), prob_sim.item()),
                        log_file=log_file)

                    print_util(
                        'Class %d | Approx1 MRE: %.3f | Approx2 MRE: %.3f | Approx3 MRE: %.3f | Approx4 MRE: %.3f'
                        % (i, approx1_mre.item(), approx2_mre.item(), approx3_mre.item(), approx4_mre.item()),
                        log_file=log_file)
                    print_util(
                        'Class %d | Scale MRE: %.3f | Approx Out MRE: %.3f | Approx Probabilities MRE: %.3f'
                        % (i, scale_mre.item(), approx_out_mre.item(), approx_prob_mre.item()),
                        log_file=log_file)
                    print_util(
                        'Class %d | Embedding MRE: %.3f | Out MRE: %.3f | Probabilities MRE: %.3f '
                        % (i, emb_mre.item(), out_mre.item(), prob_mre.item()),
                        log_file=log_file)

                    print_util('Class %d | Soteria Embedding MSE: %e | Soteria Embedding Cosine Similarity: %.3f '
                               % (i, soteria_mse.item(), soteria_sim.item()),
                               log_file=log_file)
                else:
                    prob_mse = torch.mean(cls_rec_prob ** 2)
                    print_util(
                        'Class %d | Probabilities MSE: %e' % (i, prob_mse.item()), log_file=log_file)
                    # prob_mses.update(prob_mse.item())

        print_util('***************************************************************', log_file=log_file)
        if args.analysis:
            approx1_mses.filter(args.ratio)
            approx2_mses.filter(args.ratio)
            approx3_mses.filter(args.ratio)
            approx4_mses.filter(args.ratio)
            mtp_approx1_mses.update(approx1_mses.avg)
            mtp_approx2_mses.update(approx2_mses.avg)
            mtp_approx3_mses.update(approx3_mses.avg)
            mtp_approx4_mses.update(approx4_mses.avg)

            scale_mses.filter(args.ratio)
            approx_out_mses.filter(args.ratio)
            approx_prob_mses.filter(args.ratio)
            mtp_scale_mses.update(scale_mses.avg)
            mtp_approx_out_mses.update(approx_out_mses.avg)
            mtp_approx_prob_mses.update(approx_prob_mses.avg)

            emb_mses.filter(args.ratio)
            out_mses.filter(args.ratio)
            prob_mses.filter(args.ratio)
            mtp_emb_mses.update(emb_mses.avg)
            mtp_out_mses.update(out_mses.avg)
            mtp_prob_mses.update(prob_mses.avg)

            emb_sims.filter(args.ratio)
            out_sims.filter(args.ratio)
            prob_sims.filter(args.ratio)
            mtp_emb_sims.update(emb_sims.avg)
            mtp_out_sims.update(out_sims.avg)
            mtp_prob_sims.update(prob_sims.avg)

            soteria_sims.filter(args.ratio)
            soteria_mses.filter(args.ratio)
            mtp_soteria_sims.update(soteria_sims.avg)
            mtp_soteria_mses.update(soteria_mses.avg)

            approx1_mres.filter(args.ratio)
            approx2_mres.filter(args.ratio)
            approx3_mres.filter(args.ratio)
            approx4_mres.filter(args.ratio)
            mtp_approx1_mres.update(approx1_mres.avg)
            mtp_approx2_mres.update(approx2_mres.avg)
            mtp_approx3_mres.update(approx3_mres.avg)
            mtp_approx4_mres.update(approx4_mres.avg)

            scale_mres.filter(args.ratio)
            approx_out_mres.filter(args.ratio)
            approx_prob_mres.filter(args.ratio)
            mtp_scale_mres.update(scale_mres.avg)
            mtp_approx_out_mres.update(approx_out_mres.avg)
            mtp_approx_prob_mres.update(approx_prob_mres.avg)

            emb_mres.filter(args.ratio)
            out_mres.filter(args.ratio)
            prob_mres.filter(args.ratio)
            mtp_emb_mres.update(emb_mres.avg)
            mtp_out_mres.update(out_mres.avg)
            mtp_prob_mres.update(prob_mres.avg)

            print_util(
                'Avg Approx1 MSE: %e | Avg Approx2 MSE: %e | Avg Approx3 MSE: %e | Avg Approx4 MSE: %e'
                % (approx1_mses.avg, approx2_mses.avg, approx3_mses.avg, approx4_mses.avg),
                log_file=log_file)
            print_util(
                'Avg Scale MSE: %e | Avg Approx Out MSE: %e | Avg Approx Probabilities MSE: %e'
                % (scale_mses.avg, approx_out_mses.avg, approx_prob_mses.avg),
                log_file=log_file)
            print_util(
                'Avg Embedding MSE: %e | Avg Out MSE: %e | Avg Probabilities MSE: %e'
                % (emb_mses.avg, out_mses.avg, prob_mses.avg),
                log_file=log_file)
            print_util(
                'Avg Embedding Cosine Similarity: %.3f | Avg Out Cosine Similarity: %.3f | Avg Probabilities Cosine Similarity: %.3f '
                % (emb_sims.avg, out_sims.avg, prob_sims.avg),
                log_file=log_file)

            print_util(
                'Avg Approx1 MRE: %.3f | Avg Approx2 MRE: %.3f | Avg Approx3 MRE: %.3f | Avg Approx4 MRE: %.3f'
                % (approx1_mres.avg, approx2_mres.avg, approx3_mres.avg, approx4_mres.avg),
                log_file=log_file)
            print_util(
                'Avg Scale MRE: %.3f | Avg Approx Out MRE: %.3f | Avg Approx Probabilities MRE: %.3f'
                % (scale_mres.avg, approx_out_mres.avg, approx_prob_mres.avg),
                log_file=log_file)
            print_util(
                'Avg Embedding MRE: %.3f | Avg Out MRE: %.3f | Avg Probabilities MRE: %.3f '
                % (emb_mres.avg, out_mres.avg, prob_mres.avg),
                log_file=log_file)

            print_util(
                'Avg Soteria Embedding MSE: %e | Avg Soteria Embedding Cosine Similarity: %.3f'
                % (soteria_mses.avg, soteria_sims.avg),
                log_file=log_file)

        # iLRG label recovery

        # Random assign embeddings
        # cls_rec_probs = [torch.softmax(torch.randn(args.num_classes), dim=-1) for _ in
        #                  range(args.num_classes)]

        # 1/n for probs
        if args.estimate:
            cls_rec_probs = [torch.ones(args.num_classes) / args.num_classes for _ in
                             range(args.num_classes)]

        res, metrics = get_irlg_res(cls_rec_probs=cls_rec_probs,
                                    b_grad=b_grad,
                                    gt_label=gt_label,
                                    num_classes=args.num_classes,
                                    num_images=args.num_images,
                                    log_file=log_file,
                                    simplified=args.simplified)

        _, _, rec_instances, existences, mod_rec_instances = res

        if 'custom' in args.distribution:
            mtp_0_cnt.update(rec_instances[0])
            mtp_18_cnt.update(rec_instances[18])
            mtp_92_cnt.update(rec_instances[92])

        leacc, lnacc, irec = metrics
        iLRG_leaccs.update(leacc)
        iLRG_lnaccs.update(lnacc)
        iLRG_irecs.update(irec)
        if args.compare:
            # iDLG label recovery
            idlg_acc = get_other_res(w_grad=w_grad,
                                     num_classes=args.num_classes,
                                     existences=existences,
                                     log_file=log_file,
                                     attack_method='idlg')
            iDLG_leaccs.update(idlg_acc)
            # GradientInversion label recovery
            gi_acc = get_other_res(w_grad=w_grad,
                                   num_classes=args.num_classes,
                                   existences=existences,
                                   log_file=log_file,
                                   attack_method='gi')
            GI_leaccs.update(gi_acc)
            # SVD label recovery
            svd_acc = get_other_res(w_grad=w_grad,
                                    num_classes=args.num_classes,
                                    existences=existences,
                                    log_file=log_file,
                                    attack_method='svd',
                                    num_images=args.num_images)
            SVD_leaccs.update(svd_acc)

        # Reconstruct images
        if args.rec_img:
            model.eval()
            grads = [g.to(**setup) for g in grads]
            rec_machine = rec_imgs.GradientReconstructor(model=model,
                                                         config=rec_config,
                                                         mean_std=(dm, ds),
                                                         num_images=args.num_images,
                                                         loss_thresh=args.loss_thresh,
                                                         rec_exp_dir=args.image_dir)
            labels = None
            if args.fix_labels:
                if args.gt_labels:
                    labels = gt_label
                else:
                    labels = []
                    for idx in range(args.num_classes):
                        if mod_rec_instances[idx] > 0:
                            labels.extend([idx] * mod_rec_instances[idx])
                    labels = torch.from_numpy(np.array(labels)).long().to(device)
            print(gt_label, labels)
            print_util(' ', log_file=log_file)
            print_util('Start reconstructing images', log_file=log_file)
            if not os.path.exists(args.image_dir):
                os.makedirs(args.image_dir)
            ground_truth_den = torch.clamp(gt_data * ds + dm, 0, 1)
            for j in range(args.num_images):
                gt_filename = f"gt_{j}.png"
                torchvision.utils.save_image(ground_truth_den[j:j + 1, ...],
                                             os.path.join(args.image_dir, gt_filename))
            output, stats = rec_machine.reconstruct(grads, labels,
                                                    img_shape=(channel, *img_shape),
                                                    dryrun=args.dryrun,
                                                    aux_data=valid_data)
            print_util('End reconstructing images', log_file=log_file)
            output_den = torch.clamp(output * ds + dm, 0, 1)
            # Save the resulting image
            if args.save_image and not args.dryrun:
                for j in range(args.num_images):
                    filename = f'rec_{j}.png'

                    torchvision.utils.save_image(output_den[j:j + 1, ...],
                                                 os.path.join(args.image_dir, filename))

    # Mean Results of multiple experiments
    print_util('---------------------------------------------------------------', log_file=log_file)
    opt_string = 'Mean Ours LeAcc: %.3f | Mean Ours LnAcc: %.3f | Mean Ours IRec: %.3f' % (
        iLRG_leaccs.avg, iLRG_lnaccs.avg, iLRG_irecs.avg)
    if args.compare:
        opt_string = 'Mean iDLG LeAcc: %.3f | Mean GI LeAcc: %.3f | Mean SVD LeAcc: %.3f ' % (
            iDLG_leaccs.avg, GI_leaccs.avg, SVD_leaccs.avg) + opt_string
    print_util(opt_string, log_file=log_file)
    if args.analysis:
        print_util(
            'Mean Avg Approx1 MSE: %e | Mean Avg Approx2 MSE: %e | Mean Avg Approx3 MSE: %e | Mean Avg Approx4 MSE: %e' % (
                mtp_approx1_mses.avg, mtp_approx2_mses.avg, mtp_approx3_mses.avg, mtp_approx4_mses.avg),
            log_file=log_file)
        print_util(
            'Mean Avg Approx1 MRE: %.3f | Mean Avg Approx2 MRE: %.3f | Mean Avg Approx3 MRE: %.3f | Mean Avg Approx4 MRE: %.3f' % (
                mtp_approx1_mres.avg, mtp_approx2_mres.avg, mtp_approx3_mres.avg, mtp_approx4_mres.avg),
            log_file=log_file)

        print_util('Mean Avg Scale MSE: %e | Mean Avg Approx Out MSE: %e | Mean Avg Approx Probabilities MSE: %e' % (
            mtp_scale_mses.avg, mtp_approx_out_mses.avg, mtp_approx_prob_mses.avg), log_file=log_file)
        print_util(
            'Mean Avg Scale MRE: %.3f | Mean Avg Approx Out MRE: %.3f | Mean Avg Approx Probabilities MRE: %.3f' % (
                mtp_scale_mres.avg, mtp_approx_out_mres.avg, mtp_approx_prob_mres.avg), log_file=log_file)

        print_util(
            'Mean Avg Embedding MSE: %e | Mean Avg Out MSE: %e | Mean Avg Probabilities MSE: %e' % (
                mtp_emb_mses.avg, mtp_out_mses.avg, mtp_prob_mses.avg), log_file=log_file)
        print_util('Mean Avg Embedding MRE: %.3f | Mean Avg Out MRE: %.3f | Mean Avg Probabilities MRE: %.3f' % (
            mtp_emb_mres.avg, mtp_out_mres.avg, mtp_prob_mres.avg), log_file=log_file)
        print_util(
            'Mean Avg Embedding Cosine Similarity: %.3f | Mean Avg Out Cosine Similarity: %.3f | Mean Avg Probability Cosine Similarity: %.3f' % (
                mtp_emb_sims.avg, mtp_out_sims.avg, mtp_prob_sims.avg), log_file=log_file)

        print_util(
            'Mean Avg Soteria Embedding MSE: %e | Mean Avg Soteria Embedding Cosine Similarity: %.3f ' % (
                mtp_soteria_mses.avg, mtp_soteria_sims.avg), log_file=log_file)
    if 'custom' in args.distribution:
        print_util('Mean Rec Instances: Class 0-%d, Class 18-%d, Class 92-%d' % (
            mtp_0_cnt.avg, mtp_18_cnt.avg, mtp_92_cnt.avg), log_file=log_file)
