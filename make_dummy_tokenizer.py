import sentencepiece as spm
spm.SentencePieceTrainer.train(input="dummy_file", model_prefix='dummy_tokenizer/tokenizer', vocab_size=51200, byte_fallback=True)
