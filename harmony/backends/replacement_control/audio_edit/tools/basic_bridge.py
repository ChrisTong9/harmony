"""Base execution without importing another arm's planning or audio evidence."""
from pathlib import Path
import json,os,subprocess,tempfile
SERVICE_ROOT=Path('/home/ztong/agent/evaluation/mmae/harmony_basic_execution_v372_20261023')

def recover(ctx, original, llm=None):
    root=Path('./audio_edit/outputs');root.mkdir(parents=True,exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix='basic-route-',dir=root)).resolve()
    request=folder/'request.json';response=folder/'response.json'
    request.write_text(json.dumps(dict(original_instruction=ctx.user_prompt,input_audio=str(Path(original).resolve()),output_dir=str(folder)),ensure_ascii=False,indent=2))
    try:
        subprocess.run(['/home/ztong/agent/.venvs/mixer/bin/python',str(SERVICE_ROOT/'basic_cli.py'),str(request),str(response)],check=True,timeout=3600)
        result=json.loads(response.read_text());assert Path(result['output_audio']).is_file()
        ctx.output_audio=result['output_audio'];ctx.params.update(result['params']);ctx.warnings.extend(result['warnings'])
        ctx.execution_status=result['execution_status'];ctx.execution_error=result['execution_error']
        ctx.params['basic_execution_source_digest']=result['common_source_digest']
    except Exception as exc:
        ctx.output_audio=original;ctx.params.pop('execution_outcome',None)
        ctx.params['basic_execution_delivery']=dict(directory=str(folder),note=f'{type(exc).__name__}: {exc}')
        from audio_edit.utils.execution_outcome import annotate_outcome
        ctx.fallback_used=False;annotate_outcome(ctx,original)
    ctx.fallback_used=False
    return ctx
