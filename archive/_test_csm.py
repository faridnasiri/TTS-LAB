import sys
sys.path.insert(0, "/opt/models/csm")
try:
    from models import Model, ModelArgs
    print("ModelArgs fields:", [f.name for f in ModelArgs.__dataclass_fields__.values()])
    m = Model.from_pretrained("sesame/csm-1b")
    print("from_pretrained OK:", type(m))
except Exception as e:
    print("FAIL:", e)
    import traceback; traceback.print_exc()
