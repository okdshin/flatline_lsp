import argparse
from typing import Any, Dict, Optional, List
import threading

from lsprotocol import types as lsp

from pygls.server import LanguageServer

import torch
from transformers import (AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizer, PretrainedConfig, StoppingCriteria)
from transformers.modeling_outputs import CausalLMOutput

import infer


class LlamaCppConfig(PretrainedConfig):  # type: ignore
    model_type: str = "llama_cpp"


class LlamaCppCausalLM(PreTrainedModel):
    def __init__(self, model_name, vocab_size, config: LlamaCppConfig, n_threads: int):
        super().__init__(config)
        self.vocab_size = vocab_size

        # self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.plamo_cpp_model = infer.load_model_from_file(model_name, n_threads)

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
        logits = torch.from_numpy(self.plamo_cpp_model.calc_next_token_logits(
            input_ids.numpy(), self.vocab_size))
        return CausalLMOutput(
            loss=None,
            logits=logits,
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
        suffix_text = self.tokenizer.decode(input_ids[0][-self.stop_tokens_len:])
        return suffix_text.endswith(self.stop_word)


class StopCutoffCompletion(StoppingCriteria):
    def __init__(self, latest_completion_id: List[int], latest_completion_id_lock: threading.Lock, completion_id: int, lang_server: LanguageServer):
        super().__init__()
        self.latest_completion_id = latest_completion_id
        self.latest_completion_id_lock = latest_completion_id_lock
        self.completion_id = completion_id
        self.lang_server = lang_server

    def __call__(self, *args, **kwargs) -> bool:
        with self.latest_completion_id_lock:
            if self.latest_completion_id[0] != self.completion_id:
                self.lang_server.show_message(f"stop-cutoff-completion {self.completion_id}", lsp.MessageType.Info)
                return True
            else:
                return False


class LanguageModelForCompletion:
    def __init__(self, lang_server: LanguageServer, model_name: str, max_new_tokens: int, n_threads: int):
        self.lang_server = lang_server

        assert model_name.endswith(".gguf")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Salesforce/codegen25-7b-multi", trust_remote_code=True)
        self.model = LlamaCppCausalLM(model_name=model_name, vocab_size=len(self.tokenizer),
                                      config=LlamaCppConfig(), n_threads=n_threads)
        self.max_new_tokens = max_new_tokens

        self.latest_completion_id_lock = threading.Lock()
        self.computing_resource_lock = threading.Lock()

        self.latest_completion_id = [0]

        self.stop_word = StopWord("\n", tokenizer=self.tokenizer)

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
            tokenized_prompt = self.tokenizer(text).input_ids
            generated_tokens = self.model.generate(inputs=torch.LongTensor(
                [tokenized_prompt]), max_new_tokens=self.max_new_tokens, do_sample=False,
                #stopping_criteria=[stop_cutoff_completion, self.stop_word])[0]
                stopping_criteria=[stop_cutoff_completion])[0]
            generated_text = self.tokenizer.decode(generated_tokens[len(tokenized_prompt):])
        return generated_text


lm_for_completion: Optional[LanguageModelForCompletion] = None


server = LanguageServer("flatline-lsp", "v0.0")


@server.thread()
@server.feature(
    lsp.TEXT_DOCUMENT_COMPLETION,
    lsp.CompletionOptions(trigger_characters=[
                          ".", ",", " ", "(", ")", "[", "]", "{", "}"]),
)
def completions(
        ls: LanguageServer, params: Optional[lsp.CompletionParams] = None) -> lsp.CompletionList:
    # assert lm_for_completion is not None
    document = server.workspace.get_document(params.text_document.uri)
    line_index = params.position.line
    character_index = params.position.character
    prompt = "".join(list(document.lines[max(0, line_index-15):line_index]) +
                     [document.lines[line_index][:character_index]])

    if lm_for_completion is None:
        completed_text = "<flatline_lsp_lm_for_completion is not initialized>"
    else:
        completed_text = lm_for_completion.generate_completion(text=prompt)

    return lsp.CompletionList(
        is_incomplete=True,
        items=[lsp.CompletionItem(label="(FL)"+completed_text,
                                  insert_text=completed_text,
                                  insert_text_mode=lsp.InsertTextMode.AdjustIndentation,
                                  documentation=completed_text)],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str,
                        default="/home/okada/flatline2/codegen25-7b-multi/ggml-model-Q4_K.gguf")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--n-threads", type=int, default=8)
    args = parser.parse_args()

    global lm_for_completion
    lm_for_completion = LanguageModelForCompletion(
        lang_server=server,
        model_name=args.model_name, max_new_tokens=args.max_new_tokens, n_threads=args.n_threads)

    # server.start_tcp("127.0.0.1", 8080)
    server.start_io()


if __name__ == "__main__":
    main()
