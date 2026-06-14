import sys,json,urllib.request
d=json.loads(urllib.request.urlopen("http://localhost:8002/status").read())
for e in d['engines']:
    quants = []
    for p in e['params']:
        if p['name'] == 'quant':
            quants = [o['value'] for o in p.get('options',[])]
    print(e['key'], 'loaded='+str(e['loaded']), 'quants='+str(quants))
