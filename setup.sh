#!/usr/bin/env bash
# setup.sh — Hunyuan3D Multiview Pipeline environment bootstrap
#
# Usage:
#   bash setup.sh                  # Full install (interactive pre-flight check)
#   bash setup.sh --continue 4     # Resume from step 4 (skip earlier steps)
#   bash setup.sh --yes            # Skip interactive confirmation
#   bash setup.sh --continue 4 --yes
#
# Steps:
#   0  preflight     Pre-flight resource check + summary
#   1  system-deps   System dependencies (Linux only)
#   2  miniconda     Install Miniconda into ./conda/
#   3  conda-env     Create conda env  hy3d-mv  with Python 3.10
#   4  pytorch       Install PyTorch (CUDA 12.4 on Linux, CPU on macOS)
#   5  pip-deps      Install Python dependencies from requirements.txt
#   6  clone-repos   Clone third-party repositories into third_party/
#   7  cuda-ext      Build CUDA extensions (Linux/GPU only; skipped on macOS)
#   8  models        Download model weights via scripts/download_models.py
#   9  verify        Verification checks

set -e

# ─── Colour helpers ───────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET} $*"; }
success() { echo -e "${GREEN}[OK]${RESET}   $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
header()  { echo -e "\n${BOLD}══ Step $* ══${RESET}"; }

# ─── Parse arguments ──────────────────────────────────────────────────────────
FROM_STEP=0
YES=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --continue)
            if [[ -z "$2" || ! "$2" =~ ^[0-9]+$ ]]; then
                error "--continue requires a numeric step argument (0-9)."; exit 1
            fi
            FROM_STEP="$2"; shift 2 ;;
        --yes|-y) YES=1; shift ;;
        *) error "Unknown argument: $1"; exit 1 ;;
    esac
done

run_from() { [ "$1" -ge "$FROM_STEP" ]; }

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_CONDA_DIR="${SCRIPT_DIR}/conda"
ENV_NAME="hy3d-mv"
THIRD_PARTY_DIR="${SCRIPT_DIR}/third_party"
MODELS_DIR="${SCRIPT_DIR}/models"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"

# ─── Step 0: Pre-flight ───────────────────────────────────────────────────────
if run_from 0; then
    header "0  Pre-flight check"

    OS="$(uname -s)"; ARCH="$(uname -m)"
    info "Platform: ${OS} ${ARCH}"
    if [[ "${OS}" == "Darwin" ]]; then
        warn "macOS detected — CUDA extensions and GPU stages will be SKIPPED."
        warn "Pipeline modules will be importable but GPU-only stages require RunPod."
    fi

    # git
    if command -v git &>/dev/null; then
        success "git: $(git --version)"
    else
        error "git not found. Install git and re-run."; exit 1
    fi

    # conda — check local install first, then PATH
    if [[ -x "${LOCAL_CONDA_DIR}/bin/conda" ]]; then
        success "conda (local): $("${LOCAL_CONDA_DIR}/bin/conda" --version)"
    elif command -v conda &>/dev/null; then
        success "conda (system): $(conda --version)"
    else
        warn "conda not found — will install Miniconda into ${LOCAL_CONDA_DIR} (Step 2)"
    fi

    # Linux checks
    if [[ "${OS}" == "Linux" ]]; then
        if command -v nvidia-smi &>/dev/null; then
            success "nvidia-smi: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
        else
            warn "nvidia-smi not found — CUDA extensions cannot be compiled (Step 7 will skip)"
        fi
        if command -v nvcc &>/dev/null; then
            success "nvcc: $(nvcc --version | grep 'release')"
        else
            warn "nvcc not found — install CUDA 12.4 toolkit for Step 7"
        fi
    fi

    # Disk space
    AVAIL_KB=$(df -k "${SCRIPT_DIR}" | awk 'NR==2 {print $4}')
    AVAIL_GB=$(( AVAIL_KB / 1024 / 1024 ))
    if [[ ${AVAIL_GB} -ge 27 ]]; then
        success "Disk space: ${AVAIL_GB} GB available (need ~27 GB)"
    else
        warn "Disk space: only ${AVAIL_GB} GB available — models need ~25 GB"
    fi

    # What's already present
    REPOS_CLONED=0
    [[ -d "${THIRD_PARTY_DIR}/Hunyuan3D-2.1/.git" && -d "${THIRD_PARTY_DIR}/Hunyuan3D-2/.git" && -d "${THIRD_PARTY_DIR}/MV-Adapter/.git" ]] && REPOS_CLONED=1
    MODELS_PRESENT=0
    [[ -d "${MODELS_DIR}/hunyuan3d-dit-v2-1" && -d "${MODELS_DIR}/hunyuan3d-paintpbr-v2-1" ]] && MODELS_PRESENT=1
    ENV_EXISTS=0
    if [[ -x "${LOCAL_CONDA_DIR}/bin/conda" ]]; then
        "${LOCAL_CONDA_DIR}/bin/conda" info --envs 2>/dev/null | grep -q "^${ENV_NAME}" && ENV_EXISTS=1
    elif command -v conda &>/dev/null; then
        conda info --envs 2>/dev/null | grep -q "^${ENV_NAME}" && ENV_EXISTS=1
    fi
    CONDA_ALREADY=0
    { [[ -x "${LOCAL_CONDA_DIR}/bin/conda" ]] || command -v conda &>/dev/null; } && CONDA_ALREADY=1

    echo ""
    echo -e "${BOLD}  What will be installed / skipped:${RESET}"
    printf "  %-42s %s\n" "Step 1: System deps"            "$( [[ "${OS}" == "Linux" ]] && echo 'WILL RUN (Linux)' || echo 'SKIP (macOS)' )"
    printf "  %-42s %s\n" "Step 2: Install Miniconda"      "$( [[ ${CONDA_ALREADY} -eq 1 ]] && echo 'SKIP (already found)' || echo "WILL INSTALL → ${LOCAL_CONDA_DIR}" )"
    printf "  %-42s %s\n" "Step 3: Create conda env"       "$( [[ ${ENV_EXISTS} -eq 1 ]] && echo "SKIP (${ENV_NAME} exists)" || echo "WILL CREATE ${ENV_NAME}" )"
    printf "  %-42s %s\n" "Step 4: Install PyTorch"        "$( [[ "${OS}" == "Darwin" ]] && echo 'CPU-only (macOS)' || echo 'CUDA 12.4' )"
    printf "  %-42s %s\n" "Step 5: Python dependencies"    "from requirements.txt"
    printf "  %-42s %s\n" "Step 6: Clone repos"            "$( [[ ${REPOS_CLONED} -eq 1 ]] && echo 'SKIP (already cloned)' || echo 'WILL CLONE 4 repos + install MV-Adapter' )"
    printf "  %-42s %s\n" "Step 7: Build CUDA extensions"  "$( [[ "${OS}" == "Darwin" ]] && echo 'SKIP (macOS — no CUDA)' || ( command -v nvcc &>/dev/null && echo 'WILL BUILD' || echo 'SKIP (nvcc not found)' ) )"
    printf "  %-42s %s\n" "Step 8: Download model weights" "$( [[ ${MODELS_PRESENT} -eq 1 ]] && echo 'SKIP (already present)' || echo 'WILL DOWNLOAD ~27 GB' )"
    printf "  %-42s %s\n" "Step 9: Verification"           "WILL RUN"
    echo ""

    if [[ ${YES} -eq 0 ]]; then
        read -rp "Continue? [Y/n] " REPLY
        REPLY="${REPLY:-Y}"
        if [[ ! "${REPLY}" =~ ^[Yy]$ ]]; then
            info "Aborted by user."; exit 0
        fi
    fi
fi

# ─── Step 1: System dependencies ─────────────────────────────────────────────
if run_from 1; then
    header "1  System dependencies"
    if [[ "$(uname -s)" == "Linux" ]]; then
        info "Installing system packages via apt..."
        apt-get update -qq
        apt-get install -y --no-install-recommends \
            build-essential git wget curl unzip \
            libgl1-mesa-glx libglib2.0-0 \
            ninja-build
        success "System packages installed"
    else
        info "macOS — skipping apt installs"
    fi
fi

# ─── Step 2: Install Miniconda ────────────────────────────────────────────────
if run_from 2; then
    header "2  Miniconda"
    if [[ -x "${LOCAL_CONDA_DIR}/bin/conda" ]]; then
        success "conda already in ${LOCAL_CONDA_DIR} — skipping install"
    elif command -v conda &>/dev/null; then
        success "conda on PATH — skipping install"
    else
        info "Downloading Miniconda installer..."
        _OS="$(uname -s)"; _ARCH="$(uname -m)"
        if [[ "${_OS}" == "Darwin" && "${_ARCH}" == "arm64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        elif [[ "${_OS}" == "Darwin" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        else
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        fi
        INSTALLER="/tmp/miniconda_installer_$$.sh"
        wget -q -O "${INSTALLER}" "${MINICONDA_URL}"
        bash "${INSTALLER}" -b -p "${LOCAL_CONDA_DIR}"
        rm -f "${INSTALLER}"
        success "Miniconda installed to ${LOCAL_CONDA_DIR}"
    fi
fi

# ─── Locate and source conda shell functions ──────────────────────────────────
# This must run outside any step guard so conda activate works for all
# remaining steps (mirrors the pattern from comfyui_install.sh).
if [[ -x "${LOCAL_CONDA_DIR}/bin/conda" ]]; then
    CONDA_BASE="${LOCAL_CONDA_DIR}"
elif command -v conda &>/dev/null; then
    CONDA_BASE="$(conda info --base)"
else
    error "conda not found — cannot continue. Run Step 2 first."; exit 1
fi

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>/dev/null || true

# ─── Step 3: Create conda env ─────────────────────────────────────────────────
if run_from 3; then
    header "3  Conda environment  ${ENV_NAME}"
    if conda info --envs | grep -q "^${ENV_NAME}"; then
        success "Conda env '${ENV_NAME}' already exists — skipping"
    else
        info "Creating conda env '${ENV_NAME}' with Python 3.10..."
        conda create -y -n "${ENV_NAME}" python=3.10
        success "Conda env '${ENV_NAME}' created"
    fi
fi

# Activate — unconditional (same pattern as comfyui_install.sh line 96)
conda activate "${ENV_NAME}"
info "Active env: ${CONDA_DEFAULT_ENV:-unknown}"

# ─── Step 4: PyTorch ──────────────────────────────────────────────────────────
if run_from 4; then
    header "4  PyTorch"
    if [[ "$(uname -s)" == "Darwin" ]]; then
        info "macOS — installing CPU-only PyTorch 2.5.1..."
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
            --index-url https://download.pytorch.org/whl/cpu
    else
        info "Linux — installing PyTorch 2.5.1 with CUDA 12.4..."
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
            --index-url https://download.pytorch.org/whl/cu124
    fi
    python -c "import torch; print(f'torch {torch.__version__}  CUDA: {torch.cuda.is_available()}')"
    success "PyTorch installed"
fi

# ─── Step 5: Python dependencies ──────────────────────────────────────────────
if run_from 5; then
    header "5  Python dependencies"
    if [[ ! -f "${REQUIREMENTS}" ]]; then
        error "requirements.txt not found at ${REQUIREMENTS}"; exit 1
    fi
    pip install -r "${REQUIREMENTS}"
    success "Python dependencies installed"
fi

# ─── Step 6: Clone repositories ───────────────────────────────────────────────
if run_from 6; then
    header "6  Third-party repositories"
    mkdir -p "${THIRD_PARTY_DIR}"

    clone_or_skip() {
        local repo_url="$1" dest="$2" label="$3"
        if [[ -d "${dest}/.git" ]]; then
            success "${label} already cloned — skipping"
        else
            info "Cloning ${label}..."
            git clone --depth 1 "${repo_url}" "${dest}"
            success "${label} cloned"
        fi
    }

    clone_or_skip \
        "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git" \
        "${THIRD_PARTY_DIR}/Hunyuan3D-2.1" \
        "Hunyuan3D-2.1"

    clone_or_skip \
        "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git" \
        "${THIRD_PARTY_DIR}/Hunyuan3D-2" \
        "Hunyuan3D-2"

    clone_or_skip \
        "https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1.git" \
        "${THIRD_PARTY_DIR}/ComfyUI-Hunyuan3d-2-1" \
        "ComfyUI-Hunyuan3d-2-1 (reference)"

    # MV-Adapter — canonical multiview generation stage
    clone_or_skip \
        "https://github.com/huanngzh/MV-Adapter.git" \
        "${THIRD_PARTY_DIR}/MV-Adapter" \
        "MV-Adapter (canonical_multiview stage)"

    # Install MV-Adapter as an editable Python package so `import mvadapter` works.
    # Its requirements.txt pulls in diffusers/transformers/accelerate (already present
    # from requirements.txt) so this is mostly a no-op on top of the existing env.
    MV_ADAPTER_DIR="${THIRD_PARTY_DIR}/MV-Adapter"
    if [[ -d "${MV_ADAPTER_DIR}" ]]; then
        if python -c "import mvadapter" 2>/dev/null; then
            success "mvadapter already importable — skipping install"
        else
            info "Installing MV-Adapter Python package..."
            if [[ -f "${MV_ADAPTER_DIR}/requirements.txt" ]]; then
                pip install -r "${MV_ADAPTER_DIR}/requirements.txt" --quiet
            fi
            pip install -e "${MV_ADAPTER_DIR}" --quiet
            success "mvadapter installed"
        fi
    else
        warn "MV-Adapter directory not found — canonical_multiview stage will be unavailable"
    fi
fi

# ─── Step 7: Build CUDA extensions ────────────────────────────────────────────
if run_from 7; then
    header "7  CUDA extensions"
    if [[ "$(uname -s)" == "Darwin" ]]; then
        warn "macOS — skipping CUDA extension compilation"
    elif ! command -v nvcc &>/dev/null; then
        warn "nvcc not found — skipping CUDA extension compilation"
        warn "Install CUDA 12.4 toolkit and re-run: bash setup.sh --continue 7"
    else
        export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

        RASTERIZER_DIR="${THIRD_PARTY_DIR}/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer"
        if [[ -d "${RASTERIZER_DIR}" ]]; then
            info "Building custom_rasterizer_kernel..."
            pip install --no-build-isolation -e "${RASTERIZER_DIR}"
            success "custom_rasterizer_kernel built"
        else
            warn "Rasterizer source not found — was Step 6 completed?"
        fi

        INPAINT_DIR="${THIRD_PARTY_DIR}/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer"
        if [[ -d "${INPAINT_DIR}" && -f "${INPAINT_DIR}/compile_mesh_painter.sh" ]]; then
            info "Building mesh_inpaint_processor..."
            (cd "${INPAINT_DIR}" && bash compile_mesh_painter.sh)
            success "mesh_inpaint_processor built"
        else
            warn "Inpaint source not found — was Step 6 completed?"
        fi
    fi
fi

# ─── Step 8: Download model weights ───────────────────────────────────────────
if run_from 8; then
    header "8  Model weights"
    DOWNLOAD_SCRIPT="${SCRIPT_DIR}/scripts/download_models.py"
    if [[ ! -f "${DOWNLOAD_SCRIPT}" ]]; then
        error "download_models.py not found at ${DOWNLOAD_SCRIPT}"; exit 1
    fi
    python "${DOWNLOAD_SCRIPT}" \
        --models-dir "${MODELS_DIR}" \
        --skip-existing
    success "Model download complete"
fi

# ─── Step 9: Verification ─────────────────────────────────────────────────────
if run_from 9; then
    header "9  Verification"
    FAIL=0

    PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "${PY_VER}" == "3.10" ]]; then
        success "Python ${PY_VER}"
    else
        warn "Python ${PY_VER} (expected 3.10)"
    fi

    if python -c "import torch; v=torch.__version__; assert v.startswith('2.5'), v" 2>/dev/null; then
        success "torch $(python -c 'import torch; print(torch.__version__)')"
    else
        error "PyTorch 2.5.x not importable"; FAIL=1
    fi

    if [[ "$(uname -s)" == "Linux" ]]; then
        if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            success "torch.cuda.is_available() → True"
        else
            warn "torch.cuda.is_available() → False (expected True on GPU machine)"
        fi
    fi

    for pkg in diffusers transformers trimesh xatlas rembg omegaconf huggingface_hub; do
        if python -c "import ${pkg}" 2>/dev/null; then
            success "${pkg} importable"
        else
            error "${pkg} not importable"; FAIL=1
        fi
    done

    # mvadapter — required for the optional canonical_multiview stage
    if python -c "import mvadapter" 2>/dev/null; then
        success "mvadapter importable (canonical_multiview stage ready)"
    else
        warn "mvadapter not importable — canonical_multiview stage will fail at runtime"
        warn "Fix: pip install -e third_party/MV-Adapter  (or re-run Step 6)"
    fi

    if [[ "$(uname -s)" == "Linux" ]]; then
        if python -c "import custom_rasterizer_kernel" 2>/dev/null; then
            success "custom_rasterizer_kernel compiled"
        else
            warn "custom_rasterizer_kernel not importable (run Step 7 on GPU machine)"
        fi
    fi

    for repo in Hunyuan3D-2.1 Hunyuan3D-2 MV-Adapter; do
        if [[ -d "${THIRD_PARTY_DIR}/${repo}/.git" ]]; then
            success "third_party/${repo} cloned"
        else
            if [[ "${repo}" == "MV-Adapter" ]]; then
                warn "third_party/${repo} not found — run Step 6 to clone"
            else
                warn "third_party/${repo} not found (run Step 6)"
            fi
        fi
    done

    if python -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); import src" 2>/dev/null; then
        success "src package importable"
    else
        warn "src package not importable"
    fi

    echo ""
    if [[ ${FAIL} -eq 0 ]]; then
        success "All verification checks passed."
    else
        error "Some checks failed — review output above and re-run failed steps."
        exit 1
    fi
fi

echo ""
success "setup.sh complete."
if [[ "$(uname -s)" == "Darwin" ]]; then
    echo -e "${YELLOW}Note: GPU-only stages require RunPod or a Linux GPU machine.${RESET}"
fi
echo -e "${CYAN}Note: The canonical_multiview stage uses SDXL + MV-Adapter. Its model weights"
echo -e "      (~15 GB) are NOT downloaded by default. To download them run:${RESET}"
echo -e "  python scripts/download_models.py --models mv_adapter sdxl_base sdxl_vae dpt_midas"
