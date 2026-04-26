import json
d = json.load(open("/tmp/st.json"))["models"]
ok = [k for k,v in d.items() if v["available"]]
no = [k for k,v in d.items() if not v["available"]]
print(f"Available ({len(ok)}): {', '.join(ok)}")
print(f"Unavailable ({len(no)}):")
for k in no:
    print(f"  {k}: {d[k].get('reason','')[:80]}")
