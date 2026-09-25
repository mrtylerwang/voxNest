#!/bin/bash
# voxNest Python 运行时打包脚本
# 只为 Apple Silicon macOS 而生（MLX 推理框架硬依赖 arm64）
# 下载 python-build-standalone 嵌入式 Python，打包进 Tauri 应用

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TAURI_DIR="$PROJECT_ROOT/src-tauri"
PYTHON_DIR="$TAURI_DIR/python"

# Python 版本（python-build-standalone）
PYTHON_VERSION="3.12.7"
PYTHON_BUILD_DATE="20241016"
PLATFORM="aarch64-apple-darwin"

# 前置校验：确保当前是 Apple Silicon
ARCH=$(uname -m)
OS=$(uname -s)
if [ "$OS" != "Darwin" ] || [ "$ARCH" != "arm64" ]; then
    echo "[ERROR] voxNest 仅支持 Apple Silicon macOS (arm64)"
    echo "[ERROR] 当前环境: $OS $ARCH"
    exit 1
fi

echo "[setup_python_runtime] 平台: Apple Silicon macOS ($PLATFORM)"
echo "[setup_python_runtime] Python 版本: $PYTHON_VERSION"

# 已存在则跳过
if [ -d "$PYTHON_DIR" ] && [ -x "$PYTHON_DIR/bin/python3" ]; then
    echo "[setup_python_runtime] 嵌入式 Python 已存在，跳过下载"
    "$PYTHON_DIR/bin/python3" --version
    exit 0
fi

# 下载 URL
ARCHIVE_NAME="cpython-${PYTHON_VERSION}+${PYTHON_BUILD_DATE}-${PLATFORM}-install_only.tar.gz"
GITHUB_URL="https://github.com/indygreg/python-build-standalone/releases/download/${PYTHON_BUILD_DATE}/${ARCHIVE_NAME}"
MIRROR_URLS=(
    "https://mirror.ghproxy.com/${GITHUB_URL}"
    "https://ghproxy.net/${GITHUB_URL}"
    "https://gh-proxy.com/${GITHUB_URL}"
)
TMP_FILE="/tmp/voxnest_python_runtime.tar.gz"

echo "[setup_python_runtime] 下载: $GITHUB_URL"
echo "[setup_python_runtime] 目标: $PYTHON_DIR"

# 清理旧目录
rm -rf "$PYTHON_DIR"
mkdir -p "$PYTHON_DIR"

# 下载（依次尝试直连和镜像，每个源重试3次）
download_with_retry() {
    local url="$1"
    local output="$2"
    local max_retries=3
    local retry=0

    while [ $retry -lt $max_retries ]; do
        if command -v curl &> /dev/null; then
            if curl -L --http1.1 --connect-timeout 15 --max-time 120 --retry 2 --retry-delay 2 -o "$output" "$url" 2>/dev/null; then
                if [ -s "$output" ] && tar -tzf "$output" >/dev/null 2>&1; then
                    return 0
                fi
            fi
        elif command -v wget &> /dev/null; then
            if wget --timeout=15 --tries=2 -O "$output" "$url" 2>/dev/null; then
                if [ -s "$output" ] && tar -tzf "$output" >/dev/null 2>&1; then
                    return 0
                fi
            fi
        else
            echo "[ERROR] 未找到 curl 或 wget"
            return 1
        fi
        retry=$((retry + 1))
        echo "[setup_python_runtime]   重试 $retry/$max_retries..."
        sleep 2
    done
    return 1
}

# 依次尝试直连和镜像
DOWNLOAD_SUCCESS=false
ALL_URLS=("$GITHUB_URL" "${MIRROR_URLS[@]}")

for url in "${ALL_URLS[@]}"; do
    echo "[setup_python_runtime] 尝试下载源: $url"
    if download_with_retry "$url" "$TMP_FILE"; then
        echo "[setup_python_runtime] 下载成功"
        DOWNLOAD_SUCCESS=true
        break
    fi
    echo "[setup_python_runtime] 该源失败，尝试下一个..."
done

if [ "$DOWNLOAD_SUCCESS" = false ]; then
    echo "[ERROR] 所有下载源均失败"
    echo "[ERROR] 请检查网络连接，或手动下载后放到 $TMP_FILE"
    echo "[ERROR] 下载地址: $GITHUB_URL"
    exit 1
fi

# 解压
echo "[setup_python_runtime] 解压..."
tar -xzf "$TMP_FILE" -C "$PYTHON_DIR" --strip-components=1

# 清理临时文件
rm -f "$TMP_FILE"

# 验证
if [ -x "$PYTHON_DIR/bin/python3" ]; then
    echo "[setup_python_runtime] 成功!"
    "$PYTHON_DIR/bin/python3" --version
    echo "[setup_python_runtime] 大小: $(du -sh "$PYTHON_DIR" | cut -f1)"
else
    echo "[ERROR] Python 解压失败，未找到 bin/python3"
    ls -la "$PYTHON_DIR"
    exit 1
fi

echo "[setup_python_runtime] 完成。首次启动应用时将自动创建 venv 并安装依赖。"
