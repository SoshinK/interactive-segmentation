from isegm.utils.exp_imports.default import *
MODEL_NAME = 'thinobject_tos_hrnet18'
from isegm.model.modifiers import LRMult

def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    # model_cfg.crop_size = (320, 480)
    model_cfg.crop_size = (512, 512)
    model_cfg.num_max_points = 24
    # checkpoint_path = '/home/konstantin_soshin/Work/interactive-segmentation/weights/last_checkpoint_tos_hrnet_short_with_edge_loss_bs17.pth' #
    checkpoint_path = 'weights/coco_lvis_h18_itermask.pth'
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    model = TOSHRNetModel(width=18, ocr_width=64, with_aux_output=True, use_leaky_relu=True,
                       use_rgb_conv=False, use_disks=True, norm_radius=5,
                       with_prev_mask=True, hrnet_lr_mult=0.01)
    model.feature_extractor.context_branch.ocr_distri_head.apply(LRMult(0.1))
    model.feature_extractor.context_branch.ocr_gather_head.apply(LRMult(0.1))
    model.feature_extractor.context_branch.conv3x3_ocr.apply(LRMult(0.1))
    model.feature_extractor.context_branch.load_state_dict(state_dict['state_dict'], strict=False)
    model.to(cfg.device)
    # model.to(cfg.device)
    # model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))
    # model.feature_extractor.context_branch.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.HRNETV2_W18)

    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = 16 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = CompositionLosses([
        BootstrappedCrossEntropyLoss(), 
        DiceLoss()
        ]) # final mask loss
    loss_cfg.instance_loss_weight = 1.0
    
    loss_cfg.instance_aux_loss = SigmoidBinaryCrossEntropyLoss()
    loss_cfg.instance_aux_loss_weight = 0.2

    loss_cfg.instances_cls_head_loss = BinaryCrossEntropyLoss(class_balance=True, average='size') # loss_lr
    loss_cfg.instances_cls_head_loss_weight = 1.0
    
    loss_cfg.instances_edges_loss = CompositionLosses([
        BinaryCrossEntropyLoss(class_balance=True, average='size'),
        DiceLoss()
        ]) # loss_edge
    loss_cfg.instances_edges_loss_weight = 1.0


    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.40)),
        HorizontalFlip(),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)

    val_augmentator = Compose([
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2)


    trainset = ThinObject5k(
        cfg.THINOBJ5K_PATH,
        split='train',
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=5000,
        # epoch_len=5000,
        stuff_prob=0.30
    )


    valset = ThinObject5k(
        cfg.THINOBJ5K_PATH,
        split='val',
        augmentator=val_augmentator,
        points_sampler=points_sampler,
        epoch_len=2000

    )

    optimizer_params = {
        'lr': 5e-4, 'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[85, 100], gamma=0.1)
    trainer = TOS_HRNet_Trainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=[(0, 5), (90, 1)],
                        image_dump_interval=3000,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=3)
    trainer.run(num_epochs=100)