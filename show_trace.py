import json
d = json.load(open('/tmp/p2.json'))
t = d.get('trace', d.get('error', 'no trace'))
for line in t.replace('\\n', '\n').split('\n'):
    print(line)
