# 实验环境与复现协议

## 软件环境

ModelNet40 实验运行于 Linux x86-64 和 NVIDIA RTX 3090。Jittor 环境使用 Python 3.8.19、Jittor 1.3.11.0、GCC 9.4、CUDA Toolkit 12.2 和 cuDNN 8。PyTorch 对照使用独立环境，版本为 PyTorch 1.11.0。以下命令的工作目录均为仓库根目录，Shell 为 Bash。[Python 依赖](../requirements/) · [PyTorch 历史版本](https://pytorch.org/get-started/previous-versions/#v1110)

```bash
python3.8 -m venv .venv-jittor
python3.8 -m venv .venv
for env in .venv-jittor .venv; do
  "$env/bin/python" -m pip install --upgrade 'pip<25.1' 'setuptools<71' wheel
done
.venv-jittor/bin/python -m pip install -r requirements/jittor.txt
.venv/bin/python -m pip install 'torch==1.11.0+cu113' 'torchvision==0.12.0+cu113' --extra-index-url https://download.pytorch.org/whl/cu113
.venv/bin/python -m pip install -r requirements/torch.txt
```

RN101 的 Jittor 编码器需要 cuDNN 8。下列命令使用 `PATH` 中的 CUDA 12 系列 `nvcc` 和 NVIDIA 提供的 cuDNN 包，在 `.cache/fewshot/jtcuda/` 建立项目内的库链接。已有 cuDNN 8 安装时，`--cudnn` 可直接传入包含其头文件和库的根目录。[cuDNN 8.9.7](https://pypi.org/project/nvidia-cudnn-cu12/8.9.7.29/)

```bash
export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
.venv-jittor/bin/python -m pip install 'nvidia-cudnn-cu12==8.9.7.29'
cudnn_dir="$(.venv-jittor/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cudnn")')"
.venv-jittor/bin/python -m scripts.setup.prepare_cuda --cuda "$CUDA_HOME" --cudnn "$cudnn_dir"
```

## 作者代码、数据与权重

分词器、BPE 词表和 PyTorch 对照取自作者仓库的固定提交，包括该版本的 Dassl3D。ModelNet40 使用 HDF5 划分，包含 9,840 个训练物体和 2,468 个测试物体。每个物体按文件顺序使用前 1,024 个点。[PointCLIP](https://github.com/ZrrSkywalker/PointCLIP/tree/8656f2c0b275962da5efa6b56c691572e564b5b0) · [ModelNet40 数据包](https://huggingface.co/datasets/Msun/modelnet40/blob/main/modelnet40_ply_hdf5_2048.zip)

```bash
mkdir -p reference/upstream .cache/downloads data weights/clip weights/adapter
git clone https://github.com/ZrrSkywalker/PointCLIP.git reference/upstream/PointCLIP
git -C reference/upstream/PointCLIP checkout --detach 8656f2c0b275962da5efa6b56c691572e564b5b0

archive=.cache/downloads/modelnet40_ply_hdf5_2048.zip
curl -fL --retry 3 -o "$archive" https://huggingface.co/datasets/Msun/modelnet40/resolve/main/modelnet40_ply_hdf5_2048.zip
echo "f01b8189281fae5790e39deb9f3eca86e446b771bdc665c6ad05f28d039b20e7  $archive" | sha256sum -c - && unzip -nq "$archive" -d data
```

RN50 用于零样本，RN101 和作者的 16-shot 适配器用于少样本。转换脚本校验权重文件的 SHA-256，导出 Jittor 使用的具名 NumPy 数组。`prepare_fewshot` 同时核对 640 个训练样本的 ID 和类别。[OpenAI CLIP 权重来源](https://github.com/openai/CLIP) · [作者的适配器权重](https://drive.google.com/file/d/1hFswVidomLdYaWZZga6RpWRJW9-JJbHZ/view)

```bash
base=https://openaipublic.blob.core.windows.net/clip/models
rn50=afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762
rn101=8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599
curl -fL --retry 3 "$base/$rn50/RN50.pt" -o weights/clip/RN50.pt
curl -fL --retry 3 "$base/$rn101/RN101.pt" -o weights/clip/RN101.pt
.venv/bin/gdown '1hFswVidomLdYaWZZga6RpWRJW9-JJbHZ' -O weights/adapter/model-best.pth.tar
.venv/bin/python -m scripts.setup.convert_rn50
.venv/bin/python -m scripts.setup.prepare_fewshot
```

## RN50 零样本评估

Jittor 和 PyTorch 独立从 XYZ 输入运行投影和图文编码。`repo` 为默认六视角等权配置，`table1` 使用论文表 1 的固定参数。每个输出目录对应一次运行。两组结果通过测试物体 ID、类别顺序和提示词逐项对齐。[六视角投影误差](../results/chair/six_views.json) · [椅子投影对照图](../results/figures/chair_projection.png)

```bash
.venv/bin/python -m scripts.projection.reference
.venv-jittor/bin/python -m scripts.projection.compare

.venv/bin/python -m scripts.pytorch.zeroshot_default --output runs/pytorch/zero_reference
.venv/bin/python -m scripts.pytorch.zeroshot_table1 --output-dir runs/pytorch/zero_table1_reference
for protocol in repo table1; do
  .venv-jittor/bin/python -m scripts.zeroshot.evaluate --protocol "$protocol" --output "runs/jittor/zero/$protocol"
  .venv-jittor/bin/python -m scripts.zeroshot.compare "runs/jittor/zero/$protocol"
done
```

## RN101 少样本训练与评估

每个物体生成十张 224 × 224 深度图。每档支持集取自同一份 16-shot 名单，按类取前 K 个物体。两框架共享冻结特征、批次顺序和每轮增强（种子 1000–1249），各自初始化和训练适配器。批大小为 32，训练 250 轮。优化器为 SGD，动量 0.9，权重衰减 0.0005，标签平滑系数 0.2。首轮学习率为 10⁻⁵，随后从 0.01 按余弦日程衰减。报告末轮模型在全部 2,468 个测试物体上的 Top-1 和 Top-5。[训练 ID](../manifests/modelnet40_16shot_seed2.json) · [十视角特征对照](../results/chair/ten_views.json)

```bash
POINTCLIP_RUN_DIR=runs/jittor/fewshot/16shot bash scripts/fewshot/run.sh all 0
for k in 1 2 4 8; do
  CUDA_VISIBLE_DEVICES=0 bash scripts/fewshot/run_shot.sh "$k" "runs/jittor/multishot/${k}shot"
done
for k in 1 2 4 8 16; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m scripts.pytorch.train_adapter \
    --shots "$k" --output "runs/pytorch/paired/${k}shot"
done
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 .venv/bin/python -m scripts.pytorch.fewshot_raw \
  --trained-root runs/pytorch/paired --output runs/pytorch/fewshot_raw
```

`run.sh all` 同时生成十视角椅子对照、作者权重和无适配器的测试结果，并缓存每轮增强后的冻结特征（约 3.1 GiB）。多个 GPU 编号对应并行编码任务。`scripts.pytorch.fewshot_raw` 使用作者的投影和图像编码器，从原始点云重新评估五个 PyTorch 适配器。[各组实验结果](../results/all_protocols.csv)

发布的[逐物体排名](../results/predictions.csv)包含每档两种框架预测的前五个类别。`python tools/verify_results.py` 使用标准库从该文件复核五档 Top-1、Top-5 和首选类别一致数。
