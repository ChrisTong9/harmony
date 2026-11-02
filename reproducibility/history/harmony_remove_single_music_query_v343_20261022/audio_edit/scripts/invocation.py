from dataclasses import dataclass

@dataclass
class ScriptInvocation:
    script_name:str
    params:dict
