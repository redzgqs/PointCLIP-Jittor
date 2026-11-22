# PointCLIP 训练与评估

以下命令从仓库根目录运行，使用 Bash。先完成[环境、数据与权重准备](setup.md)。示例使用 GPU 0，可换成可用的卡号。训练及评测各写入自己的目录。重跑一个已有实验时，请换一个新的输出目录。

## 六视角投影验证与零样本评估

```bash
.venv/bin/python scripts/show_chair.py
.venv-jittor/bin/python scripts/compare_projection.py
.venv-jittor/bin/python scripts/eval_zero.py --protocol repo --limit 8
```

前两条读取同一把椅子 `chair/chair_0978.ply`，从 PyTorch 固定的同一组 1,024 个点生成六张参考图，再用 Jittor 比较有效像素与深度值。第三条真正运行两座 Jittor 编码器，对测试集前八个物体分类。要测试全部 2,468 个物体并比较各自从 XYZ 开始的预测，可运行下面两组命令。

```bash
.venv-jittor/bin/python scripts/eval_zero.py --protocol repo --output runs/jittor/zero/repo
.venv/bin/python scripts/torch_repo_raw.py --output runs/pytorch/zero_reference
.venv-jittor/bin/python scripts/compare_zero.py runs/jittor/zero/repo --reference runs/pytorch/zero_reference

.venv-jittor/bin/python scripts/eval_zero.py --protocol table1 --output runs/jittor/zero/table1
.venv/bin/python scripts/torch_table1.py --output-dir runs/pytorch/zero_table1_reference
.venv-jittor/bin/python scripts/compare_zero.py runs/jittor/zero/table1 --reference runs/pytorch/zero_table1_reference
```

## 十视角特征提取与适配器训练

```bash
bash scripts/run_fewshot.sh chair 0
POINTCLIP_RUN_DIR=runs/jittor/fewshot/16shot bash scripts/run_fewshot.sh all 0
```

第一条核对十视角投影、RN101 视觉特征和作者权重。第二条缓存测试特征与训练数据每一轮增强后的特征，然后自训 16-shot 适配器并做评估。第一次全量提取需要处理 640 个训练物体的 250 份增强。训练特征约占 3.1 GiB。若有四张空闲卡，可将第二条末尾的 `0` 改为 `0 1 2 3`，分摊特征提取工作。

```bash
for k in 1 2 4 8; do
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_multishot.sh "$k" "runs/jittor/multishot/${k}shot"
done
for k in 1 2 4 8 16; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/fit_cached_torch.py \
    --shots "$k" --output "runs/pytorch/paired/${k}shot"
done
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 .venv/bin/python scripts/torch_fewshot_raw.py \
  --trained-root runs/pytorch/paired --output runs/pytorch/fewshot_raw
```

这些命令复用上一步产生的冻结特征和 PyTorch 参考预测。两框架各训练自己的适配器。每类分别取同一份 16-shot 名单的前 1、2、4、8 或 16 个物体。每轮的训练特征、顺序和评测 ID 对齐，各自框架负责随机初始化及 Dropout。最后一条重新从原始点云投影并用 PyTorch 编码，测量五套 PyTorch 权重回到原作者推理流程后的表现。[训练 ID](../manifests/modelnet40_16shot_seed2.json) · [逐轮 loss](../results/loss_16shot.csv)
