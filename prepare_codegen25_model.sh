#!/usr/bin/bash
head -c 30M </dev/urandom >dummy_file
python3 make_dummy_tokenizer.py 2>&1 | grep -v 'The corpus must be encoded in utf-8.'
