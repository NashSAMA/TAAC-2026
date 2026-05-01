# 服务器环境安装

安装脚本实例 install_taac2026_env.sh:

```bash
#!/usr/bin/env bash
set -euo pipefail

ARCHIVE="${1:-taac2026-ubuntu2204-cu126.tar.gz}"
PREFIX="${2:-$HOME/envs/taac2026}"

if [ -e "$PREFIX" ]; then
  echo "Target env already exists: $PREFIX"
  echo "Please remove it manually or choose another prefix."
  exit 1
fi

mkdir -p "$PREFIX"
tar -xzf "$ARCHIVE" -C "$PREFIX"

"$PREFIX/bin/conda-unpack"

echo "Installed to: $PREFIX"
echo
echo "Activate with:"
echo "source $PREFIX/bin/activate"
echo
"$PREFIX/bin/python" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY
```

在服务器上和压缩包同目录放一个脚本,然后服务器上执行：

```bash
chmod +x install_taac2026_env.sh

./install_taac2026_env.sh

```

如果想装到指定目录：

```bash
./install_taac2026_env.sh taac2026-ubuntu2204-cu126.tar.gz /data/envs/taac2026

```

**使用环境**

```bash
source ~/envs/taac2026/bin/activate

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

```
