import sys, os
sys.path.insert(0, '/opt/arthur')
import tts_lab_shims
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

model = ParlerTTSForConditionalGeneration.from_pretrained('parler-tts/parler-tts-mini-v1').to('cuda')
tok   = AutoTokenizer.from_pretrained('parler-tts/parler-tts-mini-v1')

iids = tok('A clear female voice.', return_tensors='pt').input_ids.to('cuda')
pids = tok('Hello world.', return_tensors='pt').input_ids.to('cuda')

with torch.no_grad():
    gen = model.generate(input_ids=iids, prompt_input_ids=pids)

print('generate OK:', gen.shape if gen is not None else None)
