_base_ = './upernet_vit-b16_mln_512x512_80k_ade20k.py'
# mmseg=0.25.0
# https://github.com/facebookresearch/xcit/blob/main/semantic_segmentation/backbone/xcit.py
model = dict(
    # pretrained='/data4/chenlinwei/code/ResolutionDet/mmsegmentation/pretrained/mmseg_deit_base_patch16_224-b5f2ef4d.pth',
    # pretrained='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_base_p16_384_20220308-96dfe169.pth',
    # pretrained='/data4/chenlinwei/code/ResolutionDet/mmsegmentation/pretrained/mmseg_deit_3_base_224_21k.pth',
    pretrained='/data4/chenlinwei/code/ResolutionDet/mmsegmentation/pretrained/deit_3_base_224_21k.pth',
    backbone=dict(
        # type='VisionTransformer_Freq',
        # type='VisionTransformer',
        # with_cp=True,
        # type='DeiT3',
        # type='vit_models',
        type='vit_models_freq',
        # type='vit_models',
        img_size=512, 
        pretrain_img_size=224,
        use_simple_fpn=False,
        with_fpn=False, 

        drop_path_rate=0.15,
        # init_scale=1,
        # with_fpn=True,
        # out_indices=[11],
        output_dtype='float32',
        # output_dtype='float16',
        # out_indices=(0, 1, 2, 3),

        # arch='b',
        # img_size=224,
        # patch_size=16,
        ),
    # neck=None,
    neck=dict(
        _delete_=True,
        type='SimpleMultiLevelNeck',
        in_channels=768,
        patch_size=16,
        carafe=False),
    # test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
    )


optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=12, 
        layer_decay_rate=0.95,
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
                # 'dy_freq': dict(lr_mult=2.),
                # 'dy_freq_2': dict(lr_mult=2.),
                # 'dy_freq_channel': dict(lr_mult=10.),
                # 'dy_freq_spatial': dict(lr_mult=10.),
                'freq_scale': dict(lr_mult=2.), 
                'freq_scale_1': dict(lr_mult=2.),
                'freq_scale_2': dict(lr_mult=2.),
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
scheduler_mult = 2
data = dict(
    # samples_per_gpu=8, workers_per_gpu=8,
    samples_per_gpu=4, workers_per_gpu=4,
    )

# fp16 = None
# optimizer_config = dict(
#     grad_clip=None,
#     type='Fp16OptimizerHook',
#     coalesce=True,
#     bucket_size_mb=-1,
#     loss_scale='dynamic',
#     distributed=True
# )

runner = dict(type='IterBasedRunner', max_iters=int(80000 * scheduler_mult))
checkpoint_config = dict(by_epoch=False, interval=int(8000 * scheduler_mult), max_keep_ckpts=2)
evaluation = dict(interval=int(8000 * scheduler_mult), metric='mIoU', pre_eval=True, save_best='mIoU')
