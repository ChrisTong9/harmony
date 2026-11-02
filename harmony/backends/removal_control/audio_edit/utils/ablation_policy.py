"""Independent execution-correction switch for the frozen factorial study."""
import os
TRACE=[]
def execution_corrections_enabled(stage):
    value=os.environ.get('HARMONY_EXECUTION_CORRECTIONS','1')
    if value not in ('0','1'):
        raise ValueError('HARMONY_EXECUTION_CORRECTIONS must be 0 or 1')
    enabled=value=='1'
    TRACE.append(dict(stage=stage,enabled=enabled))
    return enabled
