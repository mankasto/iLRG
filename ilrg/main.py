import time
from .options import options

if __name__ == '__main__':
    args = options().parse_args()

    # Import the numerical stack only after argument parsing so lightweight
    # commands such as ``python -m ilrg --help`` work before optional research
    # dependencies are installed.
    from . import consts, rec_imgs
    import torchvision
    from .utils import *
    import torch.nn as nn
    from .defense import params
    from .methods import get_emb
    from .analysis import ErrorAnalysis, class_errors
    from .models import get_model
    from .datasets import get_dataset
    from .train import Classification, train, iter_train

    args.log_dir = os.path.join('results', 'runs', args.exp_name)
    args.image_dir = os.path.join('results', 'reconstructions', args.exp_name)

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

    if args.analysis:
        analysis = ErrorAnalysis(args.ratio, log_file)

    if 'custom' in args.distribution:
        mtp_0_cnt, mtp_18_cnt, mtp_92_cnt = AverageMeter(), AverageMeter(), AverageMeter()

    target_id = args.start_id

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
            analysis.start_batch()

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
                    class_stats = (cls_embeddings[i], cls_outs[i], cls_probs[i], cls_wgrad[i], cls_bgrad[i])
                    analysis.add_class(i, class_errors(model=model,
                                                       device=device,
                                                       share=len(indexes[i]) / args.num_images,
                                                       rec_emb=cls_rec_emb,
                                                       rec_prob=cls_rec_prob,
                                                       w_grad=w_grad[i],
                                                       b_grad=b_grad[i],
                                                       class_stats=class_stats))
                else:
                    analysis.add_missing_class(i, cls_rec_prob)

        print_util('***************************************************************', log_file=log_file)
        if args.analysis:
            analysis.end_batch()

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
        analysis.print_summary()
    if 'custom' in args.distribution:
        print_util('Mean Rec Instances: Class 0-%d, Class 18-%d, Class 92-%d' % (
            mtp_0_cnt.avg, mtp_18_cnt.avg, mtp_92_cnt.avg), log_file=log_file)
