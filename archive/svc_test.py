import sys, os
sys.path.insert(0, '/opt/arthur')
import tts_lab_shims
print('shims loaded')
from indextts.infer_v2 import IndexTTS2 as IndexTTS
print('IndexTTS2 OK:', IndexTTS)
from parler_tts import ParlerTTSForConditionalGeneration
from transformers.generation.configuration_utils import GenerationConfig
print('_pad_token_tensor:', hasattr(GenerationConfig, '_pad_token_tensor'))
from transformers.masking_utils import create_causal_mask
print('masking_utils OK')
