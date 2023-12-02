#!/usr/bin/bash
#set -eux

target_dir="$HOME/.flatline_lsp"
if [ -d "$target_dir" ]; then
    echo "$target_dir" is already exist
    exit 1
fi

tmp_dir=$(mktemp -d -t mytmpdir.XXXXXX)

cleanup() {
  rm -rf "$tmp_dir"
}

trap 'cleanup' EXIT

cd "$tmp_dir"

echo "Download release"
curl -L -o flatline_lsp.zip \
    https://github.com/okdshin/flatline_lsp/releases/download/v0.0.0_a1651ec/flatline_lsp_ubuntu2004_openblas.zip
unzip flatline_lsp.zip > /dev/null

echo "Download model"
mkdir -p flatline_lsp/_internal/flatline/model_data/codegen25-7b-multi
curl -L -o flatline_lsp/_internal/flatline/model_data/codegen25-7b-multi/ggml-model-Q4_K.gguf \
    https://huggingface.co/sokada/codegen25-7b-multi-gguf-with-dummy-tokenizer/resolve/main/ggml-model-Q4_K.gguf?download=true

echo "Create $target_dir"
mv flatline_lsp "$target_dir"
