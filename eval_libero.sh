export PYTHONPATH=/home/ztj/yzh/flower_vla_calvin:$PYTHONPATH
HYDRA_FULL_ERROR=1 python flower/evaluation/flower_eval_libero.py \
  checkpoint="/home/ztj/yzh/flower_vla_calvin/logs/runs/2025-09-01/13-22-18/seed_42_offline/hf_models_checkpoint_initial/bc_vla/model.safetensors"