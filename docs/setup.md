# 从新克隆的仓库运行 PointCLIP

以下命令在 Linux x86-64 的 **Bash** 中运行，起始位置为本仓库根目录。已有原始数据和权重时，可以把它们链接到下述对应位置，直接跳到校验和转换。

实验使用 Python 3.8.19、Jittor 1.3.11.0、PyTorch 1.11.0 和 RTX 3090。Jittor 编译算子时使用 GCC 9.4、CUDA Toolkit 12.2，RN101 特征提取使用 cuDNN 8。PyTorch 使用自己的 CUDA 运行库。以下安装步骤面向同类 Linux GPU 环境；辅助 Python 包的兼容范围写在 `requirements/` 中。[Jittor 安装说明](https://pypi.org/project/jittor/1.3.11.0/) · [PyTorch 1.11 官方历史版本](https://pytorch.org/get-started/previous-versions/#v1110)

## 1. 获取指定版本的作者代码

Jittor 读取这里的 CLIP 分词器及词表。PyTorch 对照还会用到作者的投影、适配器和修改版 Dassl3D。将源码放到脚本默认读取的位置。

```bash
mkdir -p reference/upstream
git clone https://github.com/ZrrSkywalker/PointCLIP.git reference/upstream/PointCLIP
git -C reference/upstream/PointCLIP checkout --detach 8656f2c0b275962da5efa6b56c691572e564b5b0
git -C reference/upstream/PointCLIP rev-parse HEAD
```

最后一行应显示上述完整提交号。[固定版本的作者仓库](https://github.com/ZrrSkywalker/PointCLIP/tree/8656f2c0b275962da5efa6b56c691572e564b5b0)

## 2. 建立两个 Python 环境

主机需提供 Python 3.8（含 `venv` 和开发头文件）、可用的 NVIDIA 驱动、C++ 编译器以及含 `nvcc`、头文件和库的完整 CUDA Toolkit。仓库中的 Bash 脚本约定 `.venv` 用于 PyTorch，`.venv-jittor` 用于 Jittor。以下直接调用对应解释器。

```bash
python3.8 -m venv .venv
python3.8 -m venv .venv-jittor
.venv/bin/python -m pip install --upgrade 'pip<25.1' 'setuptools<71' wheel
.venv-jittor/bin/python -m pip install --upgrade 'pip<25.1' 'setuptools<71' wheel
.venv/bin/python -m pip install 'torch==1.11.0+cu113' 'torchvision==0.12.0+cu113' --extra-index-url https://download.pytorch.org/whl/cu113
.venv/bin/python -m pip install -r requirements/torch.txt
.venv-jittor/bin/python -m pip install -r requirements/jittor.txt
```

对照脚本从固定 checkout 导入 `dassl`。其中训练器导入 `torch.utils.tensorboard`，数据加载器引用 `torchvision` 和 `gdown`，评估器引用 `sklearn`。下面通过同一条导入链检验 PyTorch 环境，同时确认它能访问 GPU。[作者的训练器](https://github.com/ZrrSkywalker/PointCLIP/blob/8656f2c0b275962da5efa6b56c691572e564b5b0/Dassl3D/dassl/engine/trainer.py) · [作者的评估器](https://github.com/ZrrSkywalker/PointCLIP/blob/8656f2c0b275962da5efa6b56c691572e564b5b0/Dassl3D/dassl/evaluation/evaluator.py)

```bash
PYTHONPATH=reference/upstream/PointCLIP:reference/upstream/PointCLIP/Dassl3D .venv/bin/python - <<'PY'
import torch
from dassl.config import get_cfg_default
from trainers.fewshot import Adapter, smooth_loss
from trainers.zeroshot import PointCLIP_ZS
assert torch.cuda.is_available(), "PyTorch cannot access an NVIDIA GPU"
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
```

要跑 RN101，准备与 CUDA 12 系列相配的 cuDNN 8。已有安装可将 `POINTCLIP_CUDNN_ROOT` 指向同时包含 `include/cudnn.h` 和 `lib/` 或 `lib64/` 的目录。下面演示将 NVIDIA 的 cuDNN 安装到项目的 Jittor 环境，并使用 PATH 上的 `nvcc`。若集群使用软件模块，先加载相应的 Toolkit，使 `nvcc --version` 显示预期版本。

```bash
nvcc --version
export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
.venv-jittor/bin/python -m pip install 'nvidia-cudnn-cu12==8.9.7.29'
export POINTCLIP_CUDNN_ROOT="$(.venv-jittor/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cudnn")')"
.venv-jittor/bin/python scripts/prepare_fewshot_cuda.py --cuda "$CUDA_HOME" --cudnn "$POINTCLIP_CUDNN_ROOT"
```

看到 cuDNN 版本和 `Ready` 后，项目已在 `.cache/fewshot/jtcuda` 下建立本地链接。少样本入口会读取这个配置。只做 RN50 零样本时，可先使用主机 CUDA Toolkit 运行第 4 节的八物体检查。[NVIDIA cuDNN 8.9.7 软件包](https://pypi.org/project/nvidia-cudnn-cu12/8.9.7.29/)

## 3. 安装数据与预训练权重

本实现读取 ModelNet40 的 HDF5 点云、划分文件以及将 HDF5 行映射到物体名称的 JSON 文件。配套压缩包为 435,335,111 字节。下面从已有 SHA-256 记录的公开副本下载，并在解压前核验。使用已有数据时，可直接核对目录结构和类别顺序。[作者给出的数据入口](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) · [带 SHA-256 的公开副本](https://huggingface.co/datasets/Msun/modelnet40/blob/main/modelnet40_ply_hdf5_2048.zip)

```bash
mkdir -p .cache/downloads data weights/clip weights/adapter
archive=.cache/downloads/modelnet40_ply_hdf5_2048.zip
curl -fL --retry 3 -o "$archive" https://huggingface.co/datasets/Msun/modelnet40/resolve/main/modelnet40_ply_hdf5_2048.zip
echo "f01b8189281fae5790e39deb9f3eca86e446b771bdc665c6ad05f28d039b20e7  $archive" | sha256sum -c - && unzip -nq "$archive" -d data
```

检查 `data/modelnet40_ply_hdf5_2048/` 下能直接看到 `shape_names.txt`、`train_files.txt`、`test_files.txt`、`ply_data_train0.h5`、`ply_data_test0.h5` 和 `ply_data_test_0_id2file.json` 等。完整划分有 9,840 个训练物体和 2,468 个测试物体。脚本从每个物体按文件顺序取前 1,024 个点。

RN50 用于零样本分类。RN101 和作者的 16-shot 适配器用于少样本实验。下面的下载对象和校验值与原始归档相同。[OpenAI CLIP](https://github.com/openai/CLIP) · [PointCLIP 作者的 16-shot 权重](https://drive.google.com/file/d/1hFswVidomLdYaWZZga6RpWRJW9-JJbHZ/view)

```bash
base=https://openaipublic.blob.core.windows.net/clip/models
rn50=afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762
rn101=8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599
curl -fL --retry 3 "$base/$rn50/RN50.pt" -o weights/clip/RN50.pt
curl -fL --retry 3 "$base/$rn101/RN101.pt" -o weights/clip/RN101.pt
.venv/bin/gdown '1hFswVidomLdYaWZZga6RpWRJW9-JJbHZ' -O weights/adapter/model-best.pth.tar
echo "$rn50  weights/clip/RN50.pt" | sha256sum -c -
echo "$rn101  weights/clip/RN101.pt" | sha256sum -c -
echo 'f21942b9fe5b51075f6fd853549e3f8cbcd973f4ffcbd11f23a6f347f0edc6dc  weights/adapter/model-best.pth.tar' | sha256sum -c -
```

若 Google Drive 返回配额或访问错误，可用上方作者链接下载同一文件，放入表明的路径后再校验。已有文件均核验通过后开始转换。第一个命令导出 `weights/clip/RN50.npz`；第二个命令生成 RN101、适配器的 NumPy 权重，并验证仓库附带的 640 个固定训练 ID。

```bash
.venv/bin/python scripts/convert_clip_weights.py
.venv/bin/python scripts/prepare_fewshot.py
```

## 4. 用 8 个测试物体验证环境

在项目根目录运行一次 RN50 零样本推理。成功后终端会显示 `Scored 8/8`、准确率以及本次结果目录。

```bash
.venv-jittor/bin/python scripts/eval_zero.py --protocol repo --limit 8
```

接下来可按实验指南运行完整零样本、逐视角椅子对照及五档适配器训练。完整少样本实验可在单张 GPU 上运行，250 份逐轮增强的冻结特征约占 3.1 GiB，另计数据、权重、依赖和 Jittor 编译缓存。[实验命令与输出目录](experiments.md)
