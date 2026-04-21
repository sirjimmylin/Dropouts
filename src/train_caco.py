"""
CACo Pretraining Script with Enhanced Logging and Checkpointing.

Trains the CACo (Change-Aware Contrastive Learning) model on the 10k dataset
with periodic checkpoint saving, CSV metric logging, and final model export.
"""
import os
import csv
import time
import torch
from argparse import ArgumentParser
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pl_bolts.models.self_supervised.moco.callbacks import MocoLRScheduler

from utils.histogram_callback import HistogramCallback
from datasets.seco_datamodule import (
    ChangeAwareContrastMultiAugDataModule,
    SeasonalContrastMultiAugDataModule,
    TemporalContrastMultiAugDataModule,
    SeasonalContrastBasicDataModule,
    SeasonalContrastTemporalDataModule,
)
from models.moco2_module import MocoV2
from models.ssl_online import SSLOnlineEvaluator

import warnings
warnings.simplefilter('ignore', UserWarning)


class MetricLogger(Callback):
    """Logs training metrics to CSV and prints periodic summaries."""

    def __init__(self, log_dir, print_every=10):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / 'metrics.csv'
        self.print_every = print_every
        self.epoch_losses = []
        self.epoch_accs = {0: [], 1: [], 2: []}
        self.epoch_start_time = None
        self.train_start_time = None
        self._csv_initialized = False

    def on_train_start(self, trainer, pl_module):
        self.train_start_time = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_losses = []
        self.epoch_accs = {0: [], 1: [], 2: []}
        self.epoch_start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # PL 1.1.8 outputs format varies: can be list of dicts, dict, or tensor
        try:
            if isinstance(outputs, list):
                outputs = outputs[0]
                if isinstance(outputs, list):
                    outputs = outputs[0]
            if isinstance(outputs, dict):
                loss = outputs.get('minimize', outputs.get('loss', 0))
                if hasattr(loss, 'item'):
                    loss = loss.item()
            elif hasattr(outputs, 'item'):
                loss = outputs.item()
            else:
                loss = float(outputs)
        except (KeyError, TypeError, IndexError):
            loss = 0.0
        self.epoch_losses.append(loss)

    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        epoch = trainer.current_epoch
        epoch_time = time.time() - self.epoch_start_time
        total_time = time.time() - self.train_start_time

        avg_loss = sum(self.epoch_losses) / max(len(self.epoch_losses), 1)

        # Get logged metrics from trainer
        metrics = trainer.callback_metrics
        acc0 = metrics.get('train_acc/subspace0', 0)
        acc1 = metrics.get('train_acc/subspace1', 0)
        acc2 = metrics.get('train_acc/subspace2', 0)
        if hasattr(acc0, 'item'):
            acc0 = acc0.item()
        if hasattr(acc1, 'item'):
            acc1 = acc1.item()
        if hasattr(acc2, 'item'):
            acc2 = acc2.item()

        lr = trainer.optimizers[0].param_groups[0]['lr']

        # Write to CSV
        row = {
            'epoch': epoch,
            'loss': f'{avg_loss:.4f}',
            'acc_subspace0': f'{acc0:.2f}',
            'acc_subspace1': f'{acc1:.2f}',
            'acc_subspace2': f'{acc2:.2f}',
            'lr': f'{lr:.6f}',
            'epoch_time_s': f'{epoch_time:.1f}',
            'total_time_min': f'{total_time/60:.1f}',
        }

        if not self._csv_initialized:
            # Only write header if file doesn't exist (supports resume)
            if not self.csv_path.exists():
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    writer.writeheader()
            self._csv_initialized = True

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

        # Print summary periodically
        if epoch % self.print_every == 0 or epoch == trainer.max_epochs - 1:
            remaining = (trainer.max_epochs - epoch - 1) * epoch_time
            print(f'\n[Epoch {epoch:4d}/{trainer.max_epochs}] '
                  f'loss={avg_loss:.4f} | '
                  f'acc=[{acc0:.1f}, {acc1:.1f}, {acc2:.1f}] | '
                  f'lr={lr:.5f} | '
                  f'epoch={epoch_time:.1f}s | '
                  f'elapsed={total_time/60:.1f}min | '
                  f'ETA={remaining/60:.1f}min')


class FinalModelSaver(Callback):
    """Saves the final model after training completes.

    Follows the CACo repository's release naming convention:
      {backbone}_{method}_geo_{dataset_size}_{epochs}.ckpt  (Lightning)
      {backbone}_{method}_geo_{dataset_size}_{epochs}.pth   (encoder_q state_dict)
    """

    def __init__(self, save_dir, model_name):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

    def on_train_end(self, trainer, pl_module):
        ckpt_path = self.save_dir / f'{self.model_name}.ckpt'
        trainer.save_checkpoint(str(ckpt_path))
        print(f'\nLightning checkpoint saved: {ckpt_path}')

        pth_path = self.save_dir / f'{self.model_name}.pth'
        torch.save(pl_module.encoder_q.state_dict(), str(pth_path))
        print(f'Encoder state_dict saved:    {pth_path}')


def get_experiment_name(args):
    data_name = os.path.basename(args.data_dir)
    return f'{args.base_encoder}-{data_name}-{args.data_mode}-ep{args.max_epochs}-bs{args.batch_size}-q{args.num_negatives}'


DATAMODULES = {
    'moco': SeasonalContrastBasicDataModule,
    'moco_tp': SeasonalContrastTemporalDataModule,
    'seco': SeasonalContrastMultiAugDataModule,
    'teco': TemporalContrastMultiAugDataModule,
    'caco': ChangeAwareContrastMultiAugDataModule,
}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = MocoV2.add_model_specific_args(parser)
    parser = ArgumentParser(parents=[parser], conflict_handler='resolve', add_help=False)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--data_mode', type=str, choices=list(DATAMODULES.keys()), default='caco')
    parser.add_argument('--max_epochs', type=int, default=800)
    parser.add_argument('--schedule', type=int, nargs='*', default=[500, 700])
    parser.add_argument('--save_dir', type=str, default='../checkpoints')
    parser.add_argument('--log_dir', type=str, default='../training_logs')
    parser.add_argument('--online_data_dir', type=str, default=None)
    parser.add_argument('--online_max_epochs', type=int, default=25)
    parser.add_argument('--online_val_every_n_epoch', type=int, default=25)
    parser.add_argument('-d', '--description', default='')
    args = parser.parse_args()

    experiment_name = get_experiment_name(args)
    save_dir = Path(args.save_dir) / experiment_name
    log_dir = Path(args.log_dir) / experiment_name

    print('=' * 60)
    print(f'{args.data_mode.upper()} Pretraining Configuration')
    print('=' * 60)
    print(f'  Data mode:        {args.data_mode}')
    print(f'  Encoder:          {args.base_encoder}')
    print(f'  Batch size:       {args.batch_size}')
    print(f'  Queue size:       {args.num_negatives}')
    print(f'  Embedding dim:    {args.emb_dim}')
    print(f'  Learning rate:    {args.learning_rate}')
    print(f'  LR schedule:      {args.schedule}')
    print(f'  Epochs:           {args.max_epochs}')
    print(f'  Temperature:      {args.softmax_temperature}')
    print(f'  Momentum:         {args.encoder_momentum}')
    print(f'  Workers:          {args.num_workers}')
    print(f'  Save dir:         {save_dir}')
    print(f'  Log dir:          {log_dir}')
    print(f'  Experiment:       {experiment_name}')
    print('=' * 60)

    # Data
    datamodule_cls = DATAMODULES[args.data_mode]
    datamodule = datamodule_cls(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Model
    model = MocoV2(**vars(args), emb_spaces=datamodule.num_keys, datamodule=datamodule)

    # Logger
    logger = TensorBoardLogger(
        save_dir=str(log_dir / 'tensorboard'),
        name=experiment_name
    )

    # Checkpointing: save periodically
    periodic_ckpt = ModelCheckpoint(
        filepath=str(save_dir / f'{args.data_mode}-{{epoch:04d}}'),
        period=100,
        save_top_k=-1,  # keep all periodic checkpoints
    )

    # Callbacks
    scheduler = MocoLRScheduler(initial_lr=args.learning_rate, schedule=args.schedule, max_epochs=args.max_epochs)
    online_evaluator = SSLOnlineEvaluator(
        data_dir=args.online_data_dir,
        z_dim=model.mlp_dim,
        max_epochs=args.online_max_epochs,
        check_val_every_n_epoch=args.online_val_every_n_epoch
    )
    metric_logger = MetricLogger(log_dir=str(log_dir), print_every=10)
    # Derive model_name following the paper's release format:
    # {backbone}_{method}_geo_{dataset_size}_{epochs}
    # e.g., resnet18_caco_geo_10k_400
    _data_name = os.path.basename(args.data_dir).lower()
    if '1m' in _data_name:
        _size = '1m'
    elif '100k' in _data_name:
        _size = '100k'
    elif '10k' in _data_name:
        _size = '10k'
    else:
        _size = _data_name
    model_name = f'{args.base_encoder}_{args.data_mode}_geo_{_size}_{args.max_epochs}'
    final_saver = FinalModelSaver(save_dir=str(save_dir), model_name=model_name)
    histogram_cb = HistogramCallback(what='epochs', datamodule=datamodule)

    # Trainer
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        checkpoint_callback=periodic_ckpt,
        callbacks=[scheduler, online_evaluator, histogram_cb, metric_logger, final_saver],
        max_epochs=args.max_epochs,
        weights_summary='full'
    )

    print(f'\nStarting training at {time.strftime("%Y-%m-%d %H:%M:%S")}...\n')
    trainer.fit(model, datamodule=datamodule)
    print(f'\nTraining completed at {time.strftime("%Y-%m-%d %H:%M:%S")}')
