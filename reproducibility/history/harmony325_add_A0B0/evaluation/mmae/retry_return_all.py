"""Preserve the frozen run, supply 200 outputs, then retry its 22 failures."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

WORK = Path(__file__).resolve().parents[2]
BASE = Path('/home/ztong/agent')
EVAL = BASE / 'evaluation/mmae'
OUTPUT = EVAL / 'outputs_full_features_v2'
RESULT = EVAL / 'eval_results/full_features_v2'
sys.path.insert(0, str(WORK))
from run_agent import build_agent, run_one, utc_now, write_json_atomic, validate_runtime_environment


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    paths = list((WORK / 'audio_edit').rglob('*.py')) + list((WORK / 'evaluation').rglob('*.py'))
    return {str(p.relative_to(WORK)): digest(p) for p in sorted(paths)}


def publish(manifest):
    records = manifest['records']
    metadata = json.loads((EVAL / 'eval_metadata_full.json').read_text())
    assert len(records) == 200 and len({r['sample_id'] for r in records}) == 200
    assert [r['sample_id'] for r in records] == [m['id'] for m in metadata]
    predictions = []
    executions = []
    for record, original in zip(records, metadata):
        output = Path(record['output_audio'])
        assert output.is_file() and output.stat().st_size > 44, output
        item = copy.deepcopy(original)
        assert not any(m['role'] == 'assistant' for m in item['messages'])
        item['messages'].append({'role': 'assistant', 'content': [
            {'type': 'audio', 'audio_url': str(output)}]})
        predictions.append(item)
        executions.append({k: v for k, v in record.items() if k not in {'mmae_sample', 'previous_attempt'}})
        executions[-1]['output_sha256'] = digest(output)
    write_json_atomic(OUTPUT / 'manifest_full_all.json', manifest)
    write_json_atomic(RESULT / 'metadata.json', metadata)
    write_json_atomic(RESULT / 'predictions.json', predictions)
    write_json_atomic(RESULT / 'execution_status.json', executions)


def initialize():
    if OUTPUT.exists() or RESULT.exists():
        raise FileExistsError('Refusing to overwrite a prepared return-all run')
    previous_state = json.loads((EVAL / 'eval_results/full_return_all_v1/run_state.json').read_text())
    if previous_state['status'] != 'complete':
        raise RuntimeError('Wait for the previous run before snapshotting its outputs')
    old_path = EVAL / 'outputs_full_return_all_v1/manifest_full_all.json'
    manifest = json.loads(old_path.read_text())
    RESULT.mkdir(parents=True)
    OUTPUT.mkdir(parents=True)
    provenance = {'created_at': utc_now(), 'policy': 'weak outputs retained; exception -> original input',
                  'base_manifest': str(old_path), 'base_manifest_sha256': digest(old_path),
                  'original_provenance': str(EVAL / 'eval_results/full/agent_provenance.json'),
                  'retry_source_sha256': code_hashes(), 'retry_workspace': str(WORK)}
    write_json_atomic(RESULT / 'provenance.json', provenance)
    for record in manifest['records']:
        record['previous_attempt'] = copy.deepcopy(record)
        edited = record['status'] == 'success'
        source = Path(record['output_audio'] if edited else record['input_audio'])
        destination = OUTPUT / record['our_task'] / (record['sample_id'] + '.wav')
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source, destination)
        record.update(output_audio=str(destination), expected_output_audio=str(destination),
                      output_kind='edited' if edited else 'fallback_original',
                      fallback_used=not edited, status='success' if edited else 'fallback',
                      reused_original_success=edited, retry_completed=False)
        record['warnings'] = record.get('warnings', []) if edited else [{'code': 'fallback_original', 'reason': record['error']}]
        write_json_atomic(destination.with_suffix('.json'), record)
    publish(manifest)
    edited = sum(r['status'] == 'success' for r in manifest['records'])
    write_json_atomic(RESULT / 'run_state.json', {'status': 'seeded', 'outputs': 200,
                                                'edited': edited, 'fallback': 200-edited})
    print(f'Prepared 200 outputs: {edited} prior edits, {200-edited} fallbacks to retry')


def retry():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU retry requires an active Slurm allocation')
    print(subprocess.check_output(['nvidia-smi'], text=True), flush=True)
    from dotenv import load_dotenv
    load_dotenv(BASE / 'audio_edit/.env')

    for key, env_name in [('SAMAUDIO_PYTHON', 'sam-audio'), ('MIXER_PYTHON', 'mixer'),
                          ('STABLE_AUDIO_PYTHON', 'stable-audio-3'), ('MMAUDIO_PYTHON', 'MMAudio')]:
        os.environ[key] = str(BASE / '.venvs' / env_name / 'bin/python')
    for key, script in [('SAMAUDIO_SCRIPT', 'run_sam_audio.py'), ('MIXER_SCRIPT', 'run_mix.py'),
                         ('STABLE_AUDIO_SCRIPT', 'run_stable_audio.py')]:
        os.environ[key] = str(WORK / 'audio_edit/tools' / script)
    os.environ['PYTHONPATH'] = str(WORK)
    os.chdir(WORK)
    provenance = json.loads((RESULT / 'provenance.json').read_text())
    if code_hashes() != provenance['retry_source_sha256']:
        raise RuntimeError('Retry source changed since initialization')
    manifest = json.loads((OUTPUT / 'manifest_full_all.json').read_text())
    validate_runtime_environment(('add', 'remove', 'replace', 'volume'))
    agent = build_agent()
    candidates = [r for r in manifest['records'] if not r['reused_original_success'] and not r['retry_completed']]
    candidates.sort(key=lambda r: ('CalledProcessError' not in (r.get('error') or ''), r['sample_id']))
    write_json_atomic(RESULT / 'run_state.json', {'status': 'retrying', 'job_id': os.environ['SLURM_JOB_ID']})
    for index, prior in enumerate(candidates, 1):

        fresh = run_one(agent, prior['mmae_sample'], EVAL / 'data', OUTPUT, resume=False)
        fresh.update(previous_attempt=prior['previous_attempt'], retry_completed=True,
                     reused_original_success=False, source_revision='features_v2')
        fresh['generation_source_sha256'] = provenance['retry_source_sha256']
        fresh['retry_history'] = prior.get('retry_history', [])
        if not fresh.get('output_audio'):
            raise RuntimeError(f'Valid dataset input did not yield fallback: {prior["sample_id"]}')
        prior.clear()
        prior.update(fresh)
        write_json_atomic(Path(fresh['output_audio']).with_suffix('.json'), fresh)
        publish(manifest)

    records = manifest['records']
    state = {'status': 'complete', 'finished_at': utc_now(), 'outputs': len(records),
             'edited': sum(r['status'] == 'success' for r in records),
             'fallback': sum(r['status'] == 'fallback' for r in records),
             'warning_samples': sum(bool(r.get('warnings')) for r in records)}
    write_json_atomic(RESULT / 'run_state.json', state)
    print(json.dumps(state), flush=True)


def resume_after_fix(retry_ids=None, reason='Source update'):
    state = json.loads((RESULT / 'run_state.json').read_text())
    if state.get('job_id'):
        queue = subprocess.run(['squeue', '-h', '-j', str(state['job_id'])], capture_output=True, text=True)
        if queue.stdout.strip():
            raise RuntimeError('Stop the previous retry before changing its source snapshot')
    provenance = json.loads((RESULT / 'provenance.json').read_text())
    provenance.setdefault('source_history', []).append({
        'sha256': provenance['retry_source_sha256'], 'run_state': state,
        'superseded_at': utc_now(), 'reason': reason})
    provenance['retry_source_sha256'] = code_hashes()
    write_json_atomic(RESULT / 'provenance.json', provenance)
    manifest = json.loads((OUTPUT / 'manifest_full_all.json').read_text())
    for record in manifest['records']:
        if (not record['reused_original_success'] and record['status'] == 'fallback') or record['sample_id'] in (retry_ids or []):
            if record['retry_completed']:
                record.setdefault('retry_history', []).append({
                    k: record.get(k) for k in ('status', 'error', 'started_at', 'finished_at', 'warnings')})
                log = Path(record['log'])
                if log.is_file():
                    shutil.copy2(log, log.with_suffix(f'.attempt-{len(record["retry_history"])}.log'))
            record['retry_completed'] = False
            record['reused_original_success'] = False
    publish(manifest)
    write_json_atomic(RESULT / 'run_state.json', {'status': 'ready_to_resume'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['initialize', 'retry', 'resume-after-fix'])
    parser.add_argument('--retry-ids', nargs='*', default=[])
    parser.add_argument('--reason', default='Source update')
    args = parser.parse_args()
    if args.action == 'resume-after-fix':
        resume_after_fix(args.retry_ids, args.reason)
    else:
        {'initialize': initialize, 'retry': retry}[args.action]()
