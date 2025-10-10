_base_ = './upernet_vit-b16_mln_512x512_80k_ade20k.py'

model = dict(
    pretrained='/data3/chenlinwei/code/ResolutionDet/mmsegmentation/pretrained/mmseg_deit_tiny_patch16_224-a1311bcf.pth',
    backbone=dict(
        # type='VisionTransformer_Freq',
        # with_cp=True,
        num_heads=3, embed_dims=192, drop_path_rate=0.1),
    decode_head=dict(num_classes=150, in_channels=[192, 192, 192, 192]),
    neck=None,
    auxiliary_head=dict(num_classes=150, in_channels=192))


optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
             #   'layer_scales': dict(lr_mult=1.),
            #   'dy_freq': dict(weight_decay=0.0005),
                'dy_freq': dict(lr_mult=10.),
                # 'dy_freq': dict(decay_mult=0.0),
                'dy_freq_channel': dict(lr_mult=10.),
                'dy_freq_spatial': dict(lr_mult=10.),
            #   'dy_freq': dict(weight_decay=0.01),
            #   'freq_scale': dict(weight_decay=0.0005),
                'freq_scale': dict(lr_mult=10.),
            #   'dy_freq_starrelu': dict(lr_mult=10.),
            #   'decode_head': dict(lr_mult=2.),
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(
    # samples_per_gpu=8, workers_per_gpu=8,
    samples_per_gpu=4, workers_per_gpu=4,
    )
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000, max_keep_ckpts=2)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True, save_best='mIoU')
