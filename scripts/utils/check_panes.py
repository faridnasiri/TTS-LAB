#!/usr/bin/env python3
import re, sys

with open("/tmp/page.html") as f:
    html = f.read()

panes = re.findall(r'id="pane-([^"]+)"', html)
btns = re.findall(r"selectEngine\('([^']+)'\)", html)
ps = set(panes)
bs = set(btns)

print(f"Panes: {len(panes)}, Buttons: {len(btns)}")
if ps != bs:
    print(f"Panes without buttons: {ps - bs}")
    print(f"Buttons without panes: {bs - ps}")
else:
    print("All panes have matching buttons")

# Show first 5 of each
print("\nFirst 5 panes:", sorted(panes)[:5])
print("First 5 buttons:", sorted(btns)[:5])
