"""Small real Laya checkpoints for offline integration tests."""

import json
from pathlib import Path


def make_tiny_checkpoint(path):
    import torch
    from laya.common import DecisionModel
    from laya.train import save_checkpoint
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoModel, ModernBertConfig, PreTrainedTokenizerFast

    root = Path(__file__).resolve().parents[1]
    text = (root / 'data/examples.jsonl').read_text() + ' choice question ACTIVATE [ ] '
    splitter = Whitespace()
    words = sorted({token for token, _ in splitter.pre_tokenize_str(text)})
    specials = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    vocab = {word: i for i, word in enumerate(specials + [w for w in words if w not in specials])}
    backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = splitter
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]',
                                 cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]')
    encoder_cfg = ModernBertConfig(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
                                  num_hidden_layers=1, num_attention_heads=2,
                                  max_position_embeddings=512, sliding_window=32,
                                  pad_token_id=0, cls_token_id=2, sep_token_id=3,
                                  bos_token_id=2, eos_token_id=3)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = DecisionModel(AutoModel.from_config(encoder_cfg, attn_implementation='sdpa'), head_layers=1)
    cfg = {'encoder': 'tiny-local-encoder', 'head_layers': 1, 'act_costs': {'escalate': 0.1},
           'max_len': 256, 'head_max_len': 128,
           'temperature': [1.25, 1.0, 1.0], 'temperature_by_options': {'choice_2': 1.1}}
    save_checkpoint(model, tok, cfg, str(path))
    return cfg
