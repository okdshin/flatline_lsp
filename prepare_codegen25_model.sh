#!/usr/bin/bash
head -c 30M </dev/urandom >dummy_file
python3 make_dummy_tokenizer.py
