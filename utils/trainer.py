import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar, EarlyStopping, \
    TQDMProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor
from .ema_callback import EMA, EMAModelCheckpoint


def create_ddpmtrainer(name='model_1', save_dir='logs/', checkpoint_dir='checkpoints/', precision=16,
                       max_epoch=1000, strategy='auto', is_earlystopping=False, monitor='val_loss',
                       check_val_every_n_epoch=1, gradient_clip_val=0.5):
    print(f"name: {name}, checkpoint_dir: {checkpoint_dir}, is_earlystopping: {is_earlystopping}, monitor: {monitor}")
    # 创建 TensorBoardLogger 日志记录器
    logger = TensorBoardLogger(save_dir=save_dir, name=name)
    # 创建 ModelCheckpoint 回调
    checkpoint_callback = EMAModelCheckpoint(
        dirpath=checkpoint_dir,  # 保存模型的路径
        save_last=True,  # 是否保存最后一个模型
        save_top_k=10,  # 保存最好的几个模型
        monitor=monitor,  # 监控的指标
        filename=f'{name}-epoch={{epoch}}-{{{monitor}:.4f}}',
        mode='min',  # 保存模型的模式
        every_n_train_steps=100,  # 每多少训练步保存一次模型
        save_on_train_epoch_end=True,
    )
    ema_callback = EMA(
        decay=0.995,
        apply_ema_every_n_steps=1,
        start_step=5000,  # 从1000步开始应用EMA
        evaluate_ema_weights_instead=True  # 验证时使用EMA权重
    )
    # 学习率
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # 进度条
    progress_bar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ema_callback, checkpoint_callback, progress_bar, lr_monitor]
    # 早停
    if is_earlystopping:
        earlystopping = EarlyStopping(monitor, mode="min")
        callbacks.append(earlystopping)
    # 创建 Trainer 实例
    trainer = pl.Trainer(
        max_epochs=max_epoch,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        strategy=strategy,
        check_val_every_n_epoch=check_val_every_n_epoch,
        accumulate_grad_batches=1,
    )
    return trainer


def create_trainer(name='model_1', save_dir='logs/', checkpoint_dir='checkpoints/', precision=16,
                   max_epoch=1000, max_step=1000000, strategy='auto', is_earlystopping=False, monitor='val_loss',
                   check_val_every_n_epoch=1, gradient_clip_val=0.5):
    print(f"name: {name}, checkpoint_dir: {checkpoint_dir}, is_earlystopping: {is_earlystopping}, monitor: {monitor}")
    # 创建 TensorBoardLogger 日志记录器
    logger = TensorBoardLogger(save_dir=save_dir, name=name)
    # 创建 ModelCheckpoint 回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_last=True,
        save_top_k=10,
        monitor=monitor,
        filename=f'{name}-epoch={{epoch}}-{{{monitor}:.4f}}',
        mode='min',
        every_n_epochs=100,
        save_on_train_epoch_end=True,
    )
    # 学习率
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # 进度条
    progress_bar = TQDMProgressBar(refresh_rate=1)
    callbacks = [checkpoint_callback, progress_bar, lr_monitor]
    # 早停
    if is_earlystopping:
        earlystopping = EarlyStopping(monitor, mode="min")
        callbacks.append(earlystopping)
    # 创建 Trainer 实例
    trainer = pl.Trainer(
        max_epochs=max_epoch,
        max_steps=max_step,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        strategy=strategy,
        check_val_every_n_epoch=check_val_every_n_epoch,
        accumulate_grad_batches=1,
    )
    return trainer