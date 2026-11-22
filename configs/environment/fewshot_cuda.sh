# Source from bash, or start a subprocess with bash -c 'source ... && ...'.
# The CUDA toolkit and cuDNN sources are never modified.
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    echo 'Source this file from bash: source configs/environment/fewshot_cuda.sh' >&2
    exit 2
fi
_pointclip_fs_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)" || return 1
_pointclip_fs_cuda="$_pointclip_fs_project/.cache/fewshot/jtcuda"
if [[ ! -x "$_pointclip_fs_cuda/bin/nvcc" || ! -f "$_pointclip_fs_cuda/sources.json" ]]; then
    echo 'Prepare the project prefix first: .venv-jittor/bin/python scripts/prepare_fewshot_cuda.py' >&2
    unset _pointclip_fs_project _pointclip_fs_cuda
    return 1
fi
export CUDA_HOME="$_pointclip_fs_cuda" CUDA_PATH="$_pointclip_fs_cuda"
export nvcc_path="$_pointclip_fs_cuda/bin/nvcc"
case ":${PATH-}:" in
    *":$_pointclip_fs_cuda/bin:"*) ;;
    *) export PATH="$_pointclip_fs_cuda/bin${PATH:+:$PATH}" ;;
esac
case ":${LD_LIBRARY_PATH-}:" in
    ":$_pointclip_fs_cuda/lib64:"*) ;;
    *) export LD_LIBRARY_PATH="$_pointclip_fs_cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
export conv_opt=0 use_mkl=0 use_cutt=0 use_cutlass=0 use_nccl=0
export cache_name=pointclip_fewshot_cudnn
unset _pointclip_fs_project _pointclip_fs_cuda
