# Flatline LSP

LSP code completion server with local LLM powered by llama.cpp

## Install

Please choose "openblas" or "cublas" and replace `<release_type>` with it.

- openblas: for CPU, slow
- cublas: for NVIDIA GPU, fast

### For Ubuntu20.04

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
# This command download `flatline_lsp_ubuntu2004_<release_type>.zip` and unzip it to `$HOME/.flatline_lsp`
# then download model data and setup it
curl -L https://raw.githubusercontent.com/okdshin/flatline_lsp/main/install.sh | bash -s <release_type>
```

#### Uninstall flatline_lsp

```sh
# Uninstall: Just remove `$HOME/.flatline_lsp`
rm -R $HOME/.flatline_lsp
```

## Setup

### Neovim

Add this snippet to `plugins.lua`

```lua
local setup_flatline_lsp = function()
    vim.api.nvim_create_autocmd('FileType', {
        pattern = {'cpp', 'python'},
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
