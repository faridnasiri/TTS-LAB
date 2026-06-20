"""Run after tts_lab_shims to force-fix ExtensionsTrie."""
import transformers.tokenization_utils as tku
import transformers as tf

for cls_name in ["ExtensionsTrie", "AddedToken"]:
    # Always overwrite — transformers 5.x leaves placeholders
    stub = type(cls_name, (), {
        "__init__": lambda self, *a, **kw: None,
        "__doc__": f"Stubbed for transformers 5.x compat",
    })
    setattr(tku, cls_name, stub)
    if not isinstance(getattr(tf, cls_name, None), type):
        setattr(tf, cls_name, stub)
    print(f"Fixed {cls_name}: {type(getattr(tku, cls_name))}")
