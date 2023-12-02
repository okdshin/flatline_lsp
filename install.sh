#!/usr/bin/bash
set -eu
#set -x

backend_type=""
release_version="v0.0.0_34ca709"

help_message="
Usage: ./install.sh [OPTIONS]

Options:
  -b, --backend_type      Specify backend type (e.g., 'openblas', 'cublas')
  -r, --release_version   Specify release version (e.g., 'v0.0.0_34ca709')
  -h, --help              Display this help message and exit
"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -b|--backend_type) backend_type="$2"; shift ;;
        -r|--release_version) release_version="$2"; shift ;;
        -h|--help) echo "$help_message"; exit 0 ;;
        *) echo "Error: Unknown option $1"; exit 1 ;;
    esac
    shift
done

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

echo "Download release..."
if [ "$backend_type" == "openblas" ]; then
    curl -L -o flatline_lsp.zip \
        "https://github.com/okdshin/flatline_lsp/releases/download/${release_version}/flatline_lsp_ubuntu2004_openblas.zip"
elif [ "$backend_type" == "cublas" ]; then
    curl -L -o flatline_lsp.zip \
        "https://github.com/okdshin/flatline_lsp/releases/download/${release_version}/flatline_lsp_ubuntu2004_cublas.zip"
else
    echo "$backend_type is not supported. Please specify 'openblas' or 'cublas'"
    exit 1
fi
unzip flatline_lsp.zip > /dev/null

echo "Download tokenizer..."
mkdir -p flatline_lsp/_internal/flatline/model_data/codegen25-7b-multi
curl -L -o flatline_lsp/_internal/flatline/model_data/codegen25-7b-multi/tokenization_codegen25.py \
    https://huggingface.co/sokada/codegen25-7b-multi-gguf-with-dummy-tokenizer/resolve/main/tokenization_codegen25.py?download=true
curl -L -o flatline_lsp/_internal/flatline/model_data/codegen25-7b-multi/tokenizer_config.json \
    https://huggingface.co/sokada/codegen25-7b-multi-gguf-with-dummy-tokenizer/resolve/main/tokenizer_config.json?download=true

echo "Download model..."
mkdir -p flatline_lsp/_internal/flatline/model_data/codegen25-7b-multi
curl -L -o flatline_lsp/_internal/flatline/model_data/codegen25-7b-multi/ggml-model-Q4_K.gguf \
    https://huggingface.co/sokada/codegen25-7b-multi-gguf-with-dummy-tokenizer/resolve/main/ggml-model-Q4_K.gguf?download=true

echo "Create $target_dir..."
mv flatline_lsp "$target_dir"

echo "flatline-lsp installation finished!"
