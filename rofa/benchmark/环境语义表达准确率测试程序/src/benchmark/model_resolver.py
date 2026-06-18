"""模型路径解析与自动下载（RynnBrain-8B 与 SAM2）。

设计原则
---------
**用户对模型路径零感知，零配置**：

- 所有模型一律下载到 ``<项目根>/models/`` 下，目录名固定。
- 不读任何环境变量（``HF_ENDPOINT`` / ``RYNNBRAIN_MODEL_PATH`` / ``USE_MODELSCOPE`` 等）。
- 不接受任何用户参数：路径就是路径，不允许覆盖。
- 缺失则自动下载；下载源按 ``ModelScope -> HuggingFace`` 顺序兜底
  （ModelScope 国内零配置可用，HuggingFace 海外网络可用，二者覆盖绝大多数场景）。

固定下载目标：
- ``<root>/models/RynnBrain-8B/``         （HF: Alibaba-DAMO-Academy/RynnBrain-8B）
- ``<root>/models/sam2.1-hiera-small/``   （HF: facebook/sam2.1-hiera-small）

公共入口：
- ``ensure_rynnbrain()`` -> Path
- ``ensure_sam2()``      -> Path
- ``ensure_all()``       -> dict（一次准备好两个模型）
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ===========================================================================
# 锁定：仓库 ID 与目标目录名（与 RynnBrain 官方 README、SAM2 官方一致）
# ===========================================================================

# RynnBrain-8B
RYNN_HF_REPO = "Alibaba-DAMO-Academy/RynnBrain-8B"
RYNN_MS_REPO = "DAMO_Academy/RynnBrain-8B"
RYNN_DIRNAME = "RynnBrain-8B"

# SAM2
SAM2_HF_REPO = "facebook/sam2.1-hiera-small"
SAM2_MS_REPO = "AI-ModelScope/sam2.1-hiera-small"  # ModelScope 镜像（社区维护）
SAM2_DIRNAME = "sam2.1-hiera-small"


# ===========================================================================
# 路径
# ===========================================================================

def project_root() -> Path:
    """src/benchmark/model_resolver.py -> 项目根目录。"""
    return Path(__file__).resolve().parents[2]


def models_root() -> Path:
    """所有模型权重的根目录：``<项目根>/models/``。"""
    return project_root() / "models"


def rynnbrain_dir() -> Path:
    return models_root() / RYNN_DIRNAME


def sam2_dir() -> Path:
    return models_root() / SAM2_DIRNAME


# ===========================================================================
# 完整性校验
# ===========================================================================

def _has_config_and_weights(path: Path) -> bool:
    """目录是否为合法的 HF 模型目录（含 config.json + 至少一个权重文件）。"""
    if not path.exists() or not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    for pat in ("*.safetensors", "*.bin"):
        if any(path.glob(pat)):
            return True
    return False


def is_rynnbrain_ready() -> bool:
    return _has_config_and_weights(rynnbrain_dir())


def is_sam2_ready() -> bool:
    return _has_config_and_weights(sam2_dir())


# ===========================================================================
# 下载实现：ModelScope 优先，HuggingFace 兜底
# ===========================================================================

def _ensure_pip(pkg: str) -> None:
    """缺包就 pip install。"""
    print(f"[model] 安装依赖: {pkg}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-U", pkg],
        stdout=sys.stdout, stderr=sys.stderr,
    )


def _download_via_modelscope(repo_id: str, target_dir: Path) -> Path:
    """通过 ModelScope 下载。国内零配置可用。"""
    print(f"[model] [ModelScope] 下载 {repo_id} -> {target_dir}")
    try:
        from modelscope import snapshot_download as ms_snapshot
    except ImportError:
        _ensure_pip("modelscope")
        from modelscope import snapshot_download as ms_snapshot

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    # ModelScope 默认下载到 cache_dir/<repo_namespace>/<repo_name>/，
    # 我们用一个临时缓存目录，再移动到固定位置。
    ms_cache = target_dir.parent / ".ms_cache"
    ms_cache.mkdir(parents=True, exist_ok=True)
    downloaded = Path(ms_snapshot(repo_id, cache_dir=str(ms_cache)))

    # 把内容搬到 target_dir
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.move(str(downloaded), str(target_dir))

    # 清理空的 cache 目录
    try:
        if ms_cache.exists() and not any(ms_cache.iterdir()):
            ms_cache.rmdir()
    except OSError:
        pass

    return target_dir


def _download_via_hf(repo_id: str, target_dir: Path) -> Path:
    """通过 HuggingFace Hub 下载。"""
    print(f"[model] [HuggingFace] 下载 {repo_id} -> {target_dir}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _ensure_pip("huggingface_hub")
        from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", "*.gif", "*.mp4", ".gitattributes"],
    )
    return target_dir


def _download_with_fallback(
    target_dir: Path,
    ms_repo: Optional[str],
    hf_repo: Optional[str],
    label: str,
) -> Path:
    """按 ModelScope -> HuggingFace 顺序兜底下载。"""
    print(f"[model] === 准备 {label} ===")
    print(f"[model]   目标目录: {target_dir}")
    if target_dir.exists() and not any(target_dir.iterdir()):
        # 空目录直接删，避免下载工具误判
        target_dir.rmdir()

    errors: List[str] = []

    sources: List[tuple] = []
    if ms_repo:
        sources.append(("ModelScope", _download_via_modelscope, ms_repo))
    if hf_repo:
        sources.append(("HuggingFace", _download_via_hf, hf_repo))

    for name, fn, repo in sources:
        try:
            fn(repo, target_dir)
            if _has_config_and_weights(target_dir):
                print(f"[model] ✓ {label} 下载完成 ({name})")
                return target_dir
            errors.append(f"{name}: 下载完成但目录不完整")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
            print(f"[model] ! {name} 失败: {exc}")

    raise RuntimeError(
        f"{label} 下载失败。已尝试: \n  - "
        + "\n  - ".join(errors)
        + f"\n请检查网络或手动下载到 {target_dir}"
    )


# ===========================================================================
# 公共入口
# ===========================================================================

def ensure_rynnbrain() -> Path:
    """保证 RynnBrain-8B 就绪，返回固定路径。"""
    target = rynnbrain_dir()
    if _has_config_and_weights(target):
        print(f"[model] ✓ RynnBrain-8B 已就绪: {target}")
        return target
    return _download_with_fallback(
        target_dir=target,
        ms_repo=RYNN_MS_REPO,
        hf_repo=RYNN_HF_REPO,
        label="RynnBrain-8B (~16GB)",
    )


def ensure_sam2() -> Path:
    """保证 SAM2 就绪，返回固定路径。"""
    target = sam2_dir()
    if _has_config_and_weights(target):
        print(f"[model] ✓ SAM2 已就绪: {target}")
        return target
    return _download_with_fallback(
        target_dir=target,
        ms_repo=SAM2_MS_REPO,
        hf_repo=SAM2_HF_REPO,
        label="SAM2 (sam2.1-hiera-small, ~200MB)",
    )


def ensure_all() -> Dict[str, Path]:
    """一次性准备好所有模型；常用于 download_models.py 预热。"""
    return {
        "rynnbrain": ensure_rynnbrain(),
        "sam2": ensure_sam2(),
    }
