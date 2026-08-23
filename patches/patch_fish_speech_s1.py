"""
Patches the fish-speech MAIN checkout (/opt/models/fish-speech-s1) used by the
s1mini engine (OpenAudio S1-Mini 0.5B). Two fixes, both required for the
engine container (transformers 5.15 / engine-current):

1. tokenizer.py  -- FishTokenizer falls back to the tiktoken lib when
   AutoTokenizer fails. transformers 5.x removed the tiktoken-reading slow
   tokenizer classes that 4.x used for Qwen-style repos that ship only a bare
   tokenizer.tiktoken + special_tokens.json (fishaudio/s1-mini). Without this
   the model loads but model.tokenizer stays None and every synthesis crashes
   with "'NoneType' object has no attribute 'encode'" at content_sequence.py.

2. models/dac/inference.py -- OmegaConf.register_new_resolver("eval", eval) is
   called at module import time. The resolver registry is global, so re-import
   after the fish-speech sys.modules purge (engine reload after evict, or
   switching between the two fish checkouts) crashes with "resolver 'eval' is
   already registered". Guarded to be idempotent.

Safe to re-run -- guarded by markers. Runs as root on the VM:
    sudo python3 /opt/arthur-tts-lab/patches/patch_fish_speech_s1.py
"""
import pathlib

BASE = pathlib.Path('/opt/models/fish-speech-s1')

# ---- Fix 1: FishTokenizer tiktoken fallback ---------------------------------
p = BASE / 'fish_speech/tokenizer.py'
src = p.read_text(encoding='utf-8')

if '_TiktokenTokenizer' not in src:
    old = '        self._tokenizer = AutoTokenizer.from_pretrained(model_path)'
    new = (
        '        try:\n'
        '            self._tokenizer = AutoTokenizer.from_pretrained(model_path)\n'
        '        except Exception as _tok_err:\n'
        '            # [TTS-Lab patch 2026-08-23] transformers 5.x removed the tiktoken-\n'
        '            # reading tokenizer classes that 4.x used for Qwen-style repos that\n'
        '            # ship only a bare tokenizer.tiktoken + special_tokens.json\n'
        '            # (fishaudio/s1-mini). Load via the tiktoken lib directly - the\n'
        '            # exact encoder the checkpoint was trained with.\n'
        '            logger.warning(\n'
        '                f\'AutoTokenizer failed ({_tok_err}); falling back to tiktoken loader\'\n'
        '            )\n'
        '            self._tokenizer = _TiktokenTokenizer.load(model_path)'
    )
    assert src.count(old) == 1, 'tokenizer.py anchor not unique'
    src = src.replace(old, new)
    src += '''

class _TiktokenTokenizer:
    """Minimal tiktoken-backed tokenizer for repos that ship only a bare
    tokenizer.tiktoken + special_tokens.json (transformers 5.x fallback).
    """

    def __init__(self, enc, special: dict):
        self._enc = enc
        self._special = special or {}
        self.vocab_size = len(self._special) + len(self._enc._mergeable_ranks)

    @classmethod
    def load(cls, model_path: str) -> "_TiktokenTokenizer":
        import base64

        import tiktoken

        p = Path(model_path)
        tok_file = p if p.is_file() else p / "tokenizer.tiktoken"
        sp_file = p if p.is_file() else p / "special_tokens.json"

        mergeable_ranks = {}
        with open(tok_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token_b64, rank = line.split()
                mergeable_ranks[base64.b64decode(token_b64)] = int(rank)

        special = {}
        if sp_file.exists():
            special = json.loads(sp_file.read_text(encoding="utf-8"))

        # Qwen-family pretokenize regex (byte-level BPE)
        pat_str = (
            r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"""
            r"""\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
        )
        enc = tiktoken.Encoding(
            name="s1-mini",
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special,
        )
        return cls(enc, special)

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs) -> List[int]:
        return self._enc.encode(text, allowed_special="all")

    def decode(self, tokens, **kwargs) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return self._enc.decode(list(tokens))

    def get_vocab(self) -> dict:
        return dict(self._special)

    def convert_tokens_to_ids(self, token: str) -> int:
        if token in self._special:
            return self._special[token]
        ids = self._enc.encode(token, allowed_special="all")
        return ids[0] if ids else 0

    @property
    def pad_token_id(self) -> int:
        return self._special.get("<|pad|>", self._special.get("<|end_of_text|>", 0))

    @property
    def eos_token_id(self) -> int:
        return self._special.get("<|end_of_text|>", 0)
'''
    p.write_text(src, encoding='utf-8')
    print(f'patched: {p}')
else:
    print(f'skip (already patched): {p}')

# ---- Fix 2: idempotent OmegaConf resolver registration ----------------------
p2 = BASE / 'fish_speech/models/dac/inference.py'
src2 = p2.read_text(encoding='utf-8')
old2 = 'OmegaConf.register_new_resolver("eval", eval)'
if old2 in src2:
    new2 = (
        'try:\n'
        '    OmegaConf.register_new_resolver("eval", eval)\n'
        'except ValueError:\n'
        '    # [TTS-Lab patch 2026-08-23] resolver registry is global; the sys.modules\n'
        '    # purge between the two fish-speech checkouts (or an engine reload after\n'
        '    # evict) re-runs this module import and must not crash on re-registration.\n'
        '    pass'
    )
    assert src2.count(old2) == 1, 'dac anchor not unique'
    src2 = src2.replace(old2, new2)
    p2.write_text(src2, encoding='utf-8')
    print(f'patched: {p2}')
else:
    print(f'skip (already patched): {p2}')
