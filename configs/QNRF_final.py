
gpus = (0, 1,)
log_dir = 'exp'
workers = 6
print_freq = 20
seed = 3035

network = dict(
    backbone="MobileBackbone",
    sub_arch='mobile_large',
    counter_type = 'withMOE', #'withMOE' 'baseline'
    resolution_num = [0,1,2,3],
    loss_weight = [1., 1/2, 1/4., 1/8.],
    sigma = [4],
    gau_kernel_size = 15,
    baseline_loss = False,
    pretrained_backbone=None,

    head = dict(
        type='CountingHead',
        fuse_method = 'cat',
        in_channels=96,
        stages_channel = [160, 112, 40, 24],
        inter_layer=[64,32,16],
        out_channels=1)
    )

dataset = dict(
    name='QNRF',
    root='../ProcessedData/',
    test_set='test.txt', #'train_val.txt',
    train_set='train.txt',
    loc_gt = 'test_gt_loc.txt',
    num_classes=len(network['resolution_num']),
    den_factor=100,
    extra_train_set =None
)

optimizer = dict(
    NAME='adamw',
    BASE_LR=1e-4,
    BETAS=(0.9, 0.999),
    WEIGHT_DECAY=1e-4,
    EPS= 1.0e-08,
    MOMENTUM= 0.9,
    AMSGRAD = False,
    NESTEROV= True,
    )

lr_config = dict(
    NAME='cosine',
    WARMUP_METHOD='linear',
    DECAY_EPOCHS=250,
    DECAY_RATE = 0.1,
    WARMUP_EPOCHS=10,   # the number of epochs to warmup the lr_rate
    WARMUP_LR=5.0e-07,
    MIN_LR= 1.0e-07
  )

total_epochs = 210

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

train = dict(
    counter='normal',
    image_size=(768, 768),  # height width
    route_size=(256, 256),  # height, width
    base_size=None,
    batch_size_per_gpu=8,
    shuffle=True,
    begin_epoch=0,
    end_epoch=500,
    extra_epoch=0,
    extra_lr = 0,
    #  RESUME: true
    resume_path=None,#"
    flip=True,
    multi_scale=True,
    scale_factor=(0.5, 1/0.5),
    val_span =   [-800, -600, -400, -200, -200, -100, -100],
    downsamplerate= 1,
    ignore_label= 255
)


test = dict(
    image_size=(1024, 2048),  # height, width
    base_size=3072,
    loc_base_size=3072,
    loc_threshold = 0.15,
    batch_size_per_gpu=1,
    patch_batch_size=16,
    flip_test=False,
    multi_scale=False,
    model_file = './exp/QNRF/MobileBackbone_mobile_large/QNRF_final_2024-12-17-15-14//content/drive/MyDrive/STEERER/STEERER/exp/QNRF/MobileBackbone_mobile_large/QNRF_final_2024-12-17-15-14/Ep_427_mae_84.36152942451888_mse_150.99217291674904.pth'
    # model_file = './exp/QNRF/MocHRBackbone_hrnet48/QNRF_HR_2022-10-21-01-51/Ep_359_mae_75.40686944287694_mse_134.
)

CUDNN = dict(
    BENCHMARK= True,
    DETERMINISTIC= False,
    ENABLED= True)


