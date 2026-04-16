# src/agent/tools/registry.py
_REGISTRY = {}

def register(tool):
    cid = tool.capability_id
    # 
    _REGISTRY[cid] = tool

def get_tool(capability_id: str):
    return _REGISTRY[capability_id]

def list_tools():
    return {k: getattr(v, "name", type(v).__name__) for k, v in _REGISTRY.items()}
