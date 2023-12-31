import argparse
import time
import sys
import os
import requests
from typing import Any, Dict, Optional, List
import threading
import subprocess

from lsprotocol import types as lsp

from pygls.server import LanguageServer

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    PretrainedConfig,
    StoppingCriteria,
)
from transformers.modeling_outputs import CausalLMOutput


class LlamaCppConfig(PretrainedConfig):  # type: ignore
    model_type: str = "llama_cpp"


class LlamaCppCausalLM(PreTrainedModel):
    def __init__(
        self,
        config: LlamaCppConfig,
        model_name: str,
        backend_server_bin: str,
        backend_server_host: str,
        backend_server_port: int,
        n_threads: int,
        n_gpu_layers: int,
    ):
        super().__init__(config)

        self.backend_server_bin = backend_server_bin
        self.backend_server_host = backend_server_host
        self.backend_server_port = backend_server_port
        try:
            requests.get(
                f"http://{self.backend_server_host}:{self.backend_server_port}"
            )
        except Exception:
            subprocess.Popen(
                (
                    f"{self.backend_server_bin}"
                    f" --port {self.backend_server_port}"
                    f" --model-path {model_name}"
                    f" --n-threads {n_threads}"
                    f" --n-gpu_layers {n_gpu_layers}"
                ).split(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            while True:
                try:
                    res = requests.get(
                        f"http://{self.backend_server_host}:{self.backend_server_port}"
                    )
                    if res.status_code == 200:
                        break
                except Exception:
                    pass
                time.sleep(1)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def forward(  # type: ignore
        self,
        input_ids: torch.LongTensor,
        **kwargs,
    ) -> CausalLMOutput:
        while True:
            res = requests.post(
                f"http://{self.backend_server_host}:{self.backend_server_port}/v1/calc_next_token_logits",
                json=dict(input_tokens=input_ids[0].tolist()),
            )
            if res.status_code == 200:
                break
        return CausalLMOutput(
            loss=None,
            logits=torch.FloatTensor(res.json()["next_token_logits"]).reshape(
                1, 1, 51200
            ),
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        model_inputs = {"input_ids": input_ids}
        return model_inputs


class StopWord(StoppingCriteria):
    def __init__(self, stop_word: str, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.stop_tokens_len = len(tokenizer(stop_word).input_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        suffix_text = self.tokenizer.decode(input_ids[0][-self.stop_tokens_len :])
        return suffix_text.endswith(self.stop_word)


class CountStopWord(StoppingCriteria):
    def __init__(
        self,
        stop_word: str,
        tokenizer: PreTrainedTokenizer,
        initial_token_length: int,
        stop_word_count: int,
        min_n_tokens: int,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.initial_token_length = initial_token_length
        self.stop_word_count = stop_word_count
        self.min_n_tokens = min_n_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        if len(input_ids[0]) < self.min_n_tokens:
            return False
        generated_text: str = self.tokenizer.decode(
            input_ids[0][self.initial_token_length :]
        )
        return generated_text.count(self.stop_word) >= self.stop_word_count


class StopCutoffCompletion(StoppingCriteria):
    def __init__(
        self,
        latest_completion_id: List[int],
        latest_completion_id_lock: threading.Lock,
        completion_id: int,
        lang_server: LanguageServer,
    ):
        super().__init__()
        self.latest_completion_id = latest_completion_id
        self.latest_completion_id_lock = latest_completion_id_lock
        self.completion_id = completion_id
        self.lang_server = lang_server

    def __call__(self, *args, **kwargs) -> bool:
        with self.latest_completion_id_lock:
            if self.latest_completion_id[0] != self.completion_id:
                self.lang_server.show_message(
                    f"stop-cutoff-completion {self.completion_id}", lsp.MessageType.Info
                )
                return True
            else:
                return False


class LanguageModelForCompletion:
    def __init__(
        self,
        lang_server: LanguageServer,
        max_new_tokens: int,
        min_new_tokens: int,
        max_context_lines: int,
        max_new_lines: int,
        max_context_tokens: int,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
    ):
        self.lang_server = lang_server
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        assert self.max_new_tokens >= self.min_new_tokens

        self.max_context_lines = max_context_lines
        self.max_new_lines = max_new_lines
        self.max_context_tokens = max_context_tokens

        self.tokenizer = tokenizer
        self.model = model

        self.latest_completion_id_lock = threading.Lock()
        self.computing_resource_lock = threading.Lock()

        self.latest_completion_id = [0]  # Must be list to avoid copy

        self.stop_word = StopWord(stop_word="\n", tokenizer=self.tokenizer)

    def generate_completion(self, text: str) -> str:
        with self.latest_completion_id_lock:
            self.latest_completion_id[0] += 1
            stop_cutoff_completion = StopCutoffCompletion(
                latest_completion_id=self.latest_completion_id,
                latest_completion_id_lock=self.latest_completion_id_lock,
                completion_id=self.latest_completion_id[0],
                lang_server=self.lang_server,
            )
        with self.computing_resource_lock:
            if stop_cutoff_completion():
                return "<canceled>"
            tokenized_prompt = self.tokenizer(text).input_ids[
                -self.max_context_tokens :
            ]
            count_stop_word = CountStopWord(
                stop_word="\n",
                tokenizer=self.tokenizer,
                initial_token_length=len(tokenized_prompt),
                stop_word_count=self.max_new_lines,
                min_n_tokens=len(tokenized_prompt) + self.min_new_tokens,
            )
            generated_tokens = self.model.generate(
                inputs=torch.LongTensor([tokenized_prompt]),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                # stopping_criteria=[stop_cutoff_completion, self.stop_word],
                stopping_criteria=[stop_cutoff_completion, count_stop_word],
            )[0]
            # stopping_criteria=[stop_cutoff_completion])[0]
            generated_text = self.tokenizer.decode(
                generated_tokens[len(tokenized_prompt) :]
            )
        return generated_text


lm_for_completion: Optional[LanguageModelForCompletion] = None


server = LanguageServer("flatline-lsp", "v0.0")


@server.thread()
@server.feature(
    lsp.TEXT_DOCUMENT_COMPLETION,
    lsp.CompletionOptions(
        trigger_characters=[".", ",", " ", "(", ")", "[", "]", "{", "}"]
    ),
)
def completions(
    ls: LanguageServer, params: Optional[lsp.CompletionParams] = None
) -> lsp.CompletionList:
    # assert lm_for_completion is not None
    document = ls.workspace.get_text_document(params.text_document.uri)
    line_index = params.position.line
    character_index = params.position.character
    if lm_for_completion is not None:
        max_context_lines = lm_for_completion.max_context_lines
    else:
        max_context_lines = 4
    prompt = "".join(
        list(
            document.lines[
                max(0, line_index - max_context_lines) : line_index
            ]
        )
        + [document.lines[line_index][:character_index]]
    )

    if lm_for_completion is None:
        completed_text = "<flatline_lsp_lm_for_completion is not initialized>"
    else:
        completed_text = lm_for_completion.generate_completion(text=prompt)
    # completed_lines = completed_text.split("\n")
    # completed_text_list = ["\n".join(completed_lines[:i]) for i in range(1, len(completed_lines))]
    completed_text_list = [completed_text]

    return lsp.CompletionList(
        is_incomplete=True,
        items=[
            lsp.CompletionItem(
                # label="(FL)" + completed_text.replace("\n", "\\n"),
                label="(FL)" + completed_text,
                insert_text=completed_text,
                insert_text_mode=lsp.InsertTextMode(1),  # AsIs
                documentation=completed_text,
            )
            for completed_text in completed_text_list
        ],
    )


def resource_path(relative_path: str):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--min-new-tokens", type=int, default=32)
    parser.add_argument("--max-context-lines", type=int, default=16)
    parser.add_argument("--max-new-lines", type=int, default=8)
    parser.add_argument("--max-context-tokens", type=int, default=1024)
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        help="tokenizer name or path",
        default=resource_path("./flatline/model_data/codegen25-7b-multi"),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="model name or path",
        default=resource_path(
            "./flatline/model_data/codegen25-7b-multi/ggml-model-Q4_K.gguf"
        ),
    )
    parser.add_argument(
        "--backend-server-bin",
        type=str,
        help="llm inference backend server binary path",
        default=resource_path("./flatline/backend_server/flatline-server"),
    )
    parser.add_argument(
        "--backend-server-host",
        type=str,
        help="llm inference backend server host name",
        default="localhost",
    )
    parser.add_argument(
        "--backend-server-port",
        type=int,
        help="llm inference backend server port number",
        default=57045,
    )
    parser.add_argument("--backend-server-n-threads", type=int, default=-1)
    parser.add_argument("--backend-server-n-gpu-layers", type=int, default=35)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, trust_remote_code=True
    )

    if args.model_name.endswith(".gguf"):
        model = LlamaCppCausalLM(
            config=LlamaCppConfig(),
            model_name=args.model_name,
            backend_server_bin=args.backend_server_bin,
            backend_server_host=args.backend_server_host,
            backend_server_port=args.backend_server_port,
            n_threads=args.backend_server_n_threads,
            n_gpu_layers=args.backend_server_n_gpu_layers,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, trust_remote_code=True
        )

    global lm_for_completion
    lm_for_completion = LanguageModelForCompletion(
        lang_server=server,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        max_context_lines=args.max_context_lines,
        max_new_lines=args.max_new_lines,
        max_context_tokens=args.max_context_tokens,
        tokenizer=tokenizer,
        model=model,
    )

    server.start_io()


if __name__ == "__main__":
    main()
