import re

with open('tts_lab_ui.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: The markdown code block regex is split across lines in the HTML output
# Find: .replace(/```(\w*)\n?([\s\S]*?)```/g,
# Actually the issue is in the Python source - the regex patterns contain literal newlines
# Let's find and fix them

# The pattern in the source looks like:
# html = html.replace(/```(\w*)
# ?([\s\S]*?)```/g,
# We need to fix it so the regex stays on one line

# Find the broken pattern
old = r"""  html = html.replace(/```(\w*)
?([\s\S]*?)```/g,
    '<pre><code>$2</code></pre>');"""

new = r"""  html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, '<pre><code>$2</code></pre>');"""

if old in content:
    content = content.replace(old, new)
    print('Fixed code block regex')
else:
    # Try to find with different whitespace
    idx = content.find('```(\w*)')
    if idx > 0:
        print(f'Found at {idx}')
        print(repr(content[idx:idx+100]))

with open('tts_lab_ui.py', 'w', encoding='utf-8') as f:
    f.write(content)
