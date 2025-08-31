import torch
from pathlib import Path
from tqdm import tqdm
import hydra
import wandb
from omegaconf import DictConfig

# Assuming these functions and classes are defined elsewhere in your project
from your_project_name.trainer import QChunkingTrainer
from your_project_name.utils import log_rank_0, _move_batch_to_device, _q_chunking_offline_step, _run_q_chunking_validation

def _train_q_chunking_offline_stage(cfg: DictConfig, datamodule):
    """
    Q-chunking offline stage: Train BC + RL using ONLY offline dataset.
    Both policies learn from demonstrations.
    Creates TWO separate VLA models: BC VLA + RL VLA
    """
    log_rank_0("Starting Q-chunking offline stage")

    # Initialize trainer with offline configuration (creates 2 models: BC + RL)
    device = torch.device(cfg.device)
    q_trainer = QChunkingTrainer(cfg, device, env=None)
    q_trainer.setup_models(cfg.model)  # This creates BC VLA (4 steps) + RL VLA (1 step)
    q_trainer.setup_optimizers()

    log_rank_0(f"Created BC VLA with {q_trainer.bc_vla.num_sampling_steps} sampling steps")
    log_rank_0(f"Created RL VLA with {q_trainer.rl_vla.num_sampling_steps} sampling steps")

    # Setup data
    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()['lang']

    validation_callback = None
    if hasattr(cfg, 'callbacks') and 'rollout_hf' in cfg.callbacks:
        validation_callback = hydra.utils.instantiate(cfg.callbacks.rollout_lh)
        validation_callback.device = device

    # Save initial checkpoint before any training
    initial_checkpoint_path = Path.cwd() / "checkpoint_initial.pt"
    q_trainer.save_checkpoint(initial_checkpoint_path, cfg.checkpoint_type)
    log_rank_0("Saved initial checkpoint before training")

    # If cfg.continue_training_ckpt exists, load the checkpoint
    # Note: Assuming you've already fixed the 'continue_training_ckpt' vs 'continue_training_ckpt_path' issue from the previous conversation
    if cfg.continue_training_ckpt is not None and Path(cfg.continue_training_ckpt).exists():
        q_trainer.load_checkpoint(Path(cfg.continue_training_ckpt), 'full')
        log_rank_0(f"continue training using {cfg.continue_training_ckpt}")

    # Train for configured epochs
    for epoch in range(cfg.max_epochs):
        log_rank_0(f"Offline Epoch {epoch + 1}/{cfg.max_epochs}")

        epoch_metrics = {'bc_loss': 0.0, 'q_value': 0.0, 'q_loss': 0.0, 'distill_loss': 0.0, 'rl_actor_loss': 0.0}

        # --- 修改部分开始 ---
        # 只有主进程（rank 0）才显示进度条
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{cfg.max_epochs}")
        else:
            progress_bar = train_dataloader

        for batch_idx, dataset_batch in enumerate(progress_bar):
            # Format batch for VLA
            batch = {'lang': dataset_batch}
            batch = _move_batch_to_device(batch, q_trainer.device)

            # Offline training: BC + RL both learn from dataset
            step_metrics = _q_chunking_offline_step(q_trainer, batch)

            for key, value in step_metrics.items():
                epoch_metrics[key] += value


            # 只有主进程才更新进度条的后缀信息
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                progress_bar.set_postfix({
                    'bc_loss': step_metrics['bc_loss'],
                    'q_loss': step_metrics['q_loss'],
                    'rl_actor_loss': step_metrics['rl_actor_loss']
                })
        # --- 修改部分结束 ---

        # Average metrics and log to wandb
        for key in epoch_metrics:
            epoch_metrics[key] /= len(train_dataloader)

        log_rank_0(f"Offline epoch {epoch + 1} metrics: {epoch_metrics}")

        # Log to wandb
        wandb_metrics = {f"offline/{k}": v for k, v in epoch_metrics.items()}
        wandb_metrics["offline/epoch"] = epoch + 1
        wandb.log(wandb_metrics)

        if validation_callback:
            _run_q_chunking_validation(validation_callback, q_trainer.rl_vla, epoch, "offline")

        # Save periodic checkpoints based on config save_freq
        if (epoch + 1) % cfg.q_chunking.checkpoint.save_freq == 0:
            checkpoint_path = Path.cwd() / f"checkpoint_epoch_{epoch+1}.pt"
            q_trainer.save_checkpoint(checkpoint_path, "full")
            log_rank_0(f" Saved checkpoint at epoch {epoch+1}")

    # Save final checkpoint
    checkpoint_path = Path.cwd() / "checkpoint_offline_complete.pt"
    q_trainer.save_checkpoint(checkpoint_path, cfg.checkpoint_type)
    log_rank_0("Q-chunking offline stage completed")