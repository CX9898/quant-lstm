# syntax=docker/dockerfile:1

# Reusable CUDA + PyTorch build environment for quant-lstm.
ARG BASE_IMAGE=nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE} AS native

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG TZ=UTC
ARG APT_MIRROR=
ARG PIP_INDEX_URL=
ARG PIP_TRUSTED_HOST=
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu128
ARG TORCH_PACKAGES="torch"
ARG EXTRA_PYTHON_PACKAGES="numpy onnx onnxruntime onnxscript pybind11"
ARG TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6;8.9;9.0;12.0"

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=${TZ} \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    VIRTUAL_ENV=/opt/quant-lstm-venv \
    PATH=/opt/quant-lstm-venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Ubuntu 24.04 stores its repository definitions in ubuntu.sources, while
# derived images may still use the legacy sources.list file.
RUN if [[ -n "${APT_MIRROR}" ]]; then \
        for source in \
            /etc/apt/sources.list \
            /etc/apt/sources.list.d/ubuntu.sources; do \
            if [[ -f "${source}" ]]; then \
                sed -i \
                    -e "s|http://archive.ubuntu.com|${APT_MIRROR}|g" \
                    -e "s|http://security.ubuntu.com|${APT_MIRROR}|g" \
                    "${source}"; \
            fi; \
        done; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        clang-format \
        cmake \
        curl \
        file \
        gdb \
        git \
        htop \
        jq \
        less \
        ninja-build \
        nlohmann-json3-dev \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        ripgrep \
        rsync \
        tree \
        unzip \
        vim \
        wget \
        zip \
    && rm -rf /var/lib/apt/lists/*

RUN git config --system --add safe.directory /workspace

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade \
        pip \
        setuptools \
        wheel \
        build \
        clang-format==18.1.8

RUN if [[ -n "${PIP_INDEX_URL}" ]]; then \
        python -m pip config set global.index-url "${PIP_INDEX_URL}"; \
    fi \
    && if [[ -n "${PIP_TRUSTED_HOST}" ]]; then \
        python -m pip config set global.trusted-host "${PIP_TRUSTED_HOST}"; \
    fi

COPY requirements-test.txt /tmp/quant-lstm/requirements-test.txt

RUN python -m pip install --no-cache-dir \
        --requirement /tmp/quant-lstm/requirements-test.txt \
    && cmake --version \
    && test -f /usr/share/cmake/nlohmann_json/nlohmann_jsonConfig.cmake \
    && python -c \
        "from importlib.metadata import version; print(f'jsonschema: {version(\"jsonschema\")}'); print(f'referencing: {version(\"referencing\")}')"

FROM native AS development

RUN python -m pip install --no-cache-dir --pre \
        ${TORCH_PACKAGES} \
        --index-url "${PYTORCH_INDEX_URL}" \
    && python -m pip install --no-cache-dir ${EXTRA_PYTHON_PACKAGES} \
    && python -c \
        "import torch; print(f'PyTorch: {torch.__version__}'); print(f'torch CUDA: {torch.version.cuda}')"

WORKDIR /workspace

CMD ["bash"]
