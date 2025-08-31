#export http_proxy=http://192.168.16.76:18000
#export https_proxy=http://192.168.16.76:18000
#export WANDB_MODE=offline
#export WANDB_API_KEY=99e1e93205f16418dd38e200a4fa2166be378de2
export WANDB_MODE=offline
# 运行训练脚本
export FLASH_ATTENTION_SKIP=1
export XFORMERS_FORCE_DISABLE_MEMORY_EFFICIENT_ATTN=1
echo "启动训练..."
HYDRA_FULL_ERROR=1 python /home/ztj/yzh/flower_vla_calvin/flower/training_libero.py \
  q_chunking.enabled=true \
  q_chunking.training_stage=offline \
 #q_chunking.training_stage=offline \





# # 查找最新的日志目录
# echo -e "\n查找最新的日志目录..."
# LATEST_LOG_DIR=$(find ./logs/runs -type d -name "*_offline" | sort | tail -n 1)

# if [ -z "$LATEST_LOG_DIR" ]; then
#     echo "❌ 错误: 未找到日志目录"
#     exit 1
# else
#     echo "✅ 找到最新的日志目录: $LATEST_LOG_DIR"
# fi

# # 检查初始 checkpoint 是否已保存
# echo -e "\n检查初始 checkpoint..."
# SEED_DIRS=$(find "$LATEST_LOG_DIR" -type d -name "seed_*")
# for SEED_DIR in $SEED_DIRS; do
#     INITIAL_CHECKPOINT="${SEED_DIR}/checkpoint_initial.pt"

#     if [ -f "$INITIAL_CHECKPOINT" ]; then
#         echo "✅ 成功: 初始 checkpoint 已保存到: $INITIAL_CHECKPOINT"
#         # 显示文件大小
#         ls -lh "$INITIAL_CHECKPOINT"
#     else
#         echo "❌ 错误: 未在 $SEED_DIR 中找到初始 checkpoint"
#     fi
# done

# # 检查周期性保存的 checkpoint
# echo -e "\n检查周期性保存的 checkpoint..."
# for SEED_DIR in $SEED_DIRS; do
#     PERIODIC_CHECKPOINTS="${SEED_DIR}/checkpoint_epoch_*.pt"

#     if ls $PERIODIC_CHECKPOINTS 1> /dev/null 2>&1; then
#         echo "✅ 成功: 在 $SEED_DIR 中找到周期性 checkpoint:"
#         # 显示所有周期性 checkpoint 的文件大小
#         ls -lh $PERIODIC_CHECKPOINTS
#     else
#         echo "❓ 提示: 在 $SEED_DIR 中未找到周期性 checkpoint，可能训练尚未进行到保存点"
#     fi
# done

# # 检查最终 checkpoint
# echo -e "\n检查最终 checkpoint..."
# for SEED_DIR in $SEED_DIRS; do
#     FINAL_CHECKPOINT="${SEED_DIR}/checkpoint_offline_complete.pt"

#     if [ -f "$FINAL_CHECKPOINT" ]; then
#         echo "✅ 成功: 最终 checkpoint 已保存到: $FINAL_CHECKPOINT"
#         # 显示文件大小
#         ls -lh "$FINAL_CHECKPOINT"
#     else
#         echo "❓ 提示: 在 $SEED_DIR 中未找到最终 checkpoint，训练可能尚未完成"
#     fi
# done

# echo -e "\n✅ 检查完成!"
# else
#     echo "❓ 提示: 未找到周期性 checkpoint，可能训练尚未进行到保存点"
# fi

# # 检查最终 checkpoint
# echo -e "\n检查最终 checkpoint..."
# FINAL_CHECKPOINT="${CURRENT_DIR}/checkpoint_offline_complete.pt"

# if [ -f $FINAL_CHECKPOINT ]; then
#     echo "✅ 成功: 最终 checkpoint 已保存到: $FINAL_CHECKPOINT"
#     # 显示文件大小
#     ls -lh $FINAL_CHECKPOINT
# else
#     echo "❓ 提示: 最终 checkpoint 尚未生成，训练可能尚未完成"
# fi