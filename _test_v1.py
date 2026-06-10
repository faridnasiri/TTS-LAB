#!/opt/arthur-img-env/bin/python3
"""Quick test: load v1.txt messages and print summary."""
import sys
sys.path.insert(0, '/opt/arthur-img')
from ideogram4_lab_engine import _load_ideogram_v1_messages

msgs = _load_ideogram_v1_messages('a cat sitting on a sofa in a cozy living room', '16:9')
sys_prompt = msgs[0]['content']
user_prompt = msgs[1]['content']

print(f"System prompt: {len(sys_prompt)} chars (~{len(sys_prompt)//4} tokens)")
print(f"User prompt:   {len(user_prompt)} chars (~{len(user_prompt)//4} tokens)")
print()
print("=== SYSTEM (first 300 chars) ===")
print(sys_prompt[:300])
print("...")
print()
print("=== USER ===")
print(user_prompt)
print()
print("OK — v1.txt loaded successfully")
