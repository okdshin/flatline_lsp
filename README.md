# NOTICE

Current code does not work correctly because of a problem related to https://github.com/python-lsp/python-lsp-jsonrpc/issues/5

I'm working for resolve it.

# Flatline LSP

Github Colilot-like LSP code completion server with local LLM powered by llama.cpp

## About model

Flatline LSP uses CodeGen2.5 released under Apache v2 License by Salesforce.

https://huggingface.co/sokada/codegen25-7b-multi-gguf-with-dummy-tokenizer

## Install

Please choose "openblas" or "cublas" and replace `<backend_type>` with it.

- openblas: for CPU, slow
- cublas: for NVIDIA GPU, fast

### For Ubuntu20.04 & 22.04

#### Requirements

##### openblas

Install OpenBLAS library.

```sh
sudo apt-get install libopenblas0
```

##### cublas

Install CUDA library. See https://developer.nvidia.com/cuda-toolkit

#### Install flatline_lsp

```sh
curl -L https://raw.githubusercontent.com/okdshin/flatline_lsp/main/install.sh | bash -s -- -b <backend_type>
```

#### Uninstall flatline_lsp

```sh
# Uninstall: Just remove `$HOME/.flatline_lsp`
rm -R $HOME/.flatline_lsp
```

## Editor setup

### Neovim

Add this snippet to `plugins.lua`

```lua
local setup_flatline_lsp = function()
    vim.api.nvim_create_autocmd('FileType', {
        pattern = {'cpp', 'python'}, --other languages supported by the model can be added here
        callback = function()
            vim.lsp.start({
                name = 'flatline_lsp',
                capabilities = vim.lsp.protocol.make_client_capabilities(),
                cmd = { vim.fn.expand('$HOME/.flatline_lsp/flatline_lsp') },
                root_dir = vim.fs.dirname(vim.fs.find({ '.git' }, { upward = true })[1]),
            })
        end,
    })
end
setup_flatline_lsp()
```

## Backend server

`flatline_lsp` depends on `flatline-server` (It is `$HOME/.flatline_lsp/_internal/flatline/backend_server/flatline-server`) and `flatline_lsp` automatically starts `flatline-server` when it is not started yet.

The process of `flatline-server` does not stop even when `flatline_lsp` is stopped. So if you want to stop `flatline-server` and release the resources (e.g. GPU memory), please kill the `flatline-server` process manually.

TODO prepare more sophisticated method
