import sys,json
d=json.load(sys.stdin)
for e in d['engines']:
    print(e['key'], e['loaded'], e.get('error','')[:50])
