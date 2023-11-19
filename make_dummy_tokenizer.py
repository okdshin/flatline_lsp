import sentencepiece as spm
from pathlib import Path
Path("./dummy_tokenizer").mkdir(exist_ok=True)
spm.SentencePieceTrainer.train(input="dummy_file", model_prefix='dummy_tokenizer/tokenizer', vocab_size=51200, byte_fallback=True)
