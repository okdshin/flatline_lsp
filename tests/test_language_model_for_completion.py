from unittest.mock import Mock
from typing import Any

from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutput
import torch

try:
    import sys
    from pathlib import Path
    print(str(Path(__file__).parent.parent))
    sys.path.append(str(Path(__file__).parent.parent))
    import flatline_lsp
except Exception:
    pass


class DummyModelConfig(PretrainedConfig):  # type: ignore
    model_type: str = "dummy"


class DummyModel(PreTrainedModel):
    def __init__(self, config: DummyModelConfig, vocab_size: int):
        super().__init__(config)
        self.vocab_size = vocab_size

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def forward(self, input_ids: torch.LongTensor, **kwargs) -> CausalLMOutput:
        return CausalLMOutput(
            logits=torch.randn(1, 1, self.vocab_size),
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> dict[str, Any]:
        model_inputs = {"input_ids": input_ids}
        return model_inputs


def test_language_model_for_completion():
    server = Mock()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = DummyModel(config=DummyModelConfig(), vocab_size=len(tokenizer))
    lm_for_completion = flatline_lsp.LanguageModelForCompletion(
        lang_server=server,
        max_new_tokens=32,
        tokenizer=tokenizer,
        model=model,
    )
    generated_text = lm_for_completion.generate_completion("The quick brown fox")
    assert len(generated_text) != 0
