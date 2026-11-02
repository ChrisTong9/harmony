"""Read existing execution logs only. No inference, scoring, or source mutations."""
from pathlib import Path
from collections import Counter
from statistics import mean, median
from datetime import datetime, timezone
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ABLATION = BASE / 'harmony_factorial_ablation_v325_20261021'
EXPECTED = {'add': 115, 'remove': 149, 'replace': 79, 'volume': 77}
MODEL = 'gpt-5-mini-2025-08-07'


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def dump_csv(name, rows):
    with (HERE / name).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def jsonl(path):
    if not path.is_file():
        return None
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(rows, label, task):
    out = dict(cohort=label, task=task, n=len(rows))
    fields = ['runtime_seconds', 'llm_logged_calls', 'llm_successful_calls',
              'preedit_qwen_observations', 'preedit_external_tool_calls',
              'recorded_qwen_responses', 'llm_input_tokens', 'llm_output_tokens']
    for field in fields:
        values = [row[field] for row in rows if row[field] is not None]
        out[field + '_coverage'] = len(values)
        out[field + '_mean'] = mean(values) if values else None
    out['runtime_seconds_median'] = median(row['runtime_seconds'] for row in rows)
    out['preedit_external_tool_used_cases'] = sum((r['preedit_external_tool_calls'] or 0) > 0 for r in rows)
    return out


def main():
    anchors = read(ABLATION / 'state/anchors.json')
    configs, details, hashes = {}, {}, {}
    models, responses, efforts, request_settings = Counter(), Counter(), Counter(), Counter()
    qwen_models, qwen_settings = Counter(), Counter()
    tool_names = Counter()
    problems = []

    def collect(task, root):
        root = Path(root)
        if str(root) in details:
            return details[str(root)]
        config = read(root / 'run_config.json')
        selected = {key: config.get(key) for key in [
            'openai_model', 'openai_reasoning_effort', 'perception', 'limits',
            'source_digest', 'preedit_environment']}
        selected['config_path'] = str(root / 'run_config.json')
        configs[str(root)] = selected
        assert selected['openai_model'] == MODEL
        assert selected['openai_reasoning_effort'] == 'low'
        source = root / 'state/execution_status.json'
        hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        records = read(source)
        assert len(records) == EXPECTED[task]
        assert len({r['sample_id'] for r in records}) == len(records)
        out = []
        for record in records:
            params = record.get('agent_editing_parameters') or {}
            dialogue = params.get('edit_listening_dialogue')
            folder = Path(record['worker_folder'])
            log = folder / 'openai_usage.jsonl'
            calls = jsonl(log)
            analysis = jsonl(folder / 'analysis_calls.jsonl')
            row = dict(task=task, run=root.name, sample_id=record['sample_id'],
                runtime_seconds=record['runtime_seconds'],
                llm_logged_calls=len(calls) if calls is not None else None,
                llm_successful_calls=sum(c.get('status') == 'ok' for c in calls) if calls is not None else None,
                llm_input_tokens=None, llm_output_tokens=None,
                preedit_qwen_observations=None, preedit_external_tool_calls=None,
                recorded_qwen_responses=len(analysis) if analysis is not None else None,
                llm_usage_log=str(log), execution_record=str(source))
            assert isinstance(row['runtime_seconds'], (float, int)) and row['runtime_seconds'] >= 0
            if isinstance(dialogue, dict) and isinstance(dialogue.get('usage'), dict):
                row['preedit_qwen_observations'] = dialogue['usage']['qwen_listens']
                row['preedit_external_tool_calls'] = dialogue['usage']['tool_calls']
                assert row['preedit_qwen_observations'] <= 2
                assert row['preedit_external_tool_calls'] <= 2
                assert row['preedit_external_tool_calls'] == len(dialogue.get('tool_calls', []))
                tool_names.update(call['name'] for call in dialogue.get('tool_calls', []))
            if calls is not None:
                assert calls
                usage = [c.get('usage') for c in calls if c.get('status') == 'ok']
                if all(isinstance(u, dict) for u in usage):
                    row['llm_input_tokens'] = sum(u.get('prompt_tokens', 0) for u in usage)
                    row['llm_output_tokens'] = sum(u.get('completion_tokens', 0) for u in usage)
                for call in calls:
                    models[call.get('requested_model')] += 1
                    responses[call.get('response_model', 'no_response')] += 1
                    efforts[call.get('reasoning_effort')] += 1
                    settings = {k: call.get('request', {}).get(k, 'not_explicitly_set')
                        for k in ['temperature', 'top_p', 'seed', 'max_tokens', 'max_completion_tokens']}
                    request_settings[json.dumps(settings, sort_keys=True)] += 1
                    if call.get('requested_model') != MODEL or call.get('reasoning_effort') != 'low':
                        problems.append(dict(sample_id=record['sample_id'], issue='runtime planner configuration mismatch'))
                hashes[str(log)] = hashlib.sha256(log.read_bytes()).hexdigest()
            for call in analysis or []:
                request = call.get('request', {})
                qwen_models[request.get('model')] += 1
                qwen_settings[json.dumps({k: request.get(k, 'not_explicitly_set')
                    for k in ['temperature', 'max_tokens', 'seed']}, sort_keys=True)] += 1
            out.append(row)
        details[str(root)] = out
        return out

    groups, rows, summaries = {}, [], []
    for cell in ['A0B0', 'A0B1', 'A1B0', 'A1B1']:
        cohort = 'original_matched_' + cell
        groups[cohort] = []
        for task in EXPECTED:
            root = Path(anchors[task]['run']) if cell == 'A1B1' else ABLATION / 'runs' / task / cell
            values = collect(task, root)
            groups[cohort].extend(values)
            summaries.append(summarize(values, cohort, task))
        summaries.append(summarize(groups[cohort], cohort, 'all'))
    main_roots = {task: Path(anchor['run']) for task, anchor in anchors.items()}
    main_roots['replace'] = BASE / 'harmony_replace79_integrated_v363_20261022'
    cohort = 'main_source_first_execution'
    groups[cohort] = []
    for task, root in main_roots.items():
        values = collect(task, root)
        groups[cohort].extend(values)
        summaries.append(summarize(values, cohort, task))
    summaries.append(summarize(groups[cohort], cohort, 'all'))
    for cohort, values in groups.items():
        assert len(values) == 420
        rows.extend(dict(cohort=cohort, **r) for r in values)
    assert not problems, problems
    basic_root = BASE / 'harmony_basic_execution_v370_20261023/state/cases'
    basic_requests, basic_responses, basic_efforts, basic_token_limits = Counter(), Counter(), Counter(), Counter()
    basic_count = 0
    for case in sorted(basic_root.iterdir()):
        response_path = case / 'first/plan_first_response.json'
        if not response_path.is_file():
            continue
        request_path = case / 'plan_request.json'
        request, response = read(request_path), read(response_path)
        basic_count += 1
        basic_requests[request['model']] += 1
        basic_responses[response['model']] += 1
        basic_efforts[request['reasoning_effort']] += 1
        basic_token_limits[str(request['max_completion_tokens'])] += 1
        assert response['model'] == MODEL and request['reasoning_effort'] == 'low'
        for path in [request_path, response_path]:
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert basic_count == 88
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    result = dict(at=datetime.now(timezone.utc).isoformat(), new_inference=0, new_scoring=0,
        scope='Original complete-run execution episodes. Main source reference uses ADD v25, REMOVE v302, REPLACE v363, VOLUME v57, before subsequent basic-execution/development repairs.',
        current_final_pipeline_latency_measured=False,
        runtime_definition='run_one wall clock from before input resolution through agent execution and output copy; excludes Slurm queue, Qwen service startup, evaluation, later offline repair episodes and setup/build_agent outside run_one. May include tool subprocess model loading and service contention.',
        llm_count_definition='Metered chat.completions.create invocations (successful and errors), not unobserved SDK-internal network retries. Entire original execution episode, not only correction calls.',
        preedit_count_definition='Logical pre-edit Qwen observations and separately external evidence-tool dispatches; excludes post-edit observations, normal SAM/generation/mixing, evaluator calls, SDK retries and later repairs.',
        qwen_response_definition='Responses saved by Qwen analyzer, possibly including formatting attempts; missing logs are unknown, not zero. This is not asserted to cover every post-edit backend call.',
        aggregate_definition='Means cover all recorded values; every column includes coverage. Task-weighted all row uses all 420 cases, including unsuccessful first attempts.',
        source_hashes=hashes, configurations=configs,
        actual_planner_requests=dict(models), actual_planner_response_models=dict(responses),
        actual_reasoning_effort=dict(efforts), actual_planner_sampling_fields=dict(request_settings),
        recorded_qwen_models=dict(qwen_models), recorded_qwen_sampling_fields=dict(qwen_settings),
        common_basic_execution=dict(first_plans=basic_count, requests=dict(basic_requests),
            responses=dict(basic_responses), reasoning_effort=dict(basic_efforts),
            max_completion_tokens=dict(basic_token_limits), maximum_planner_calls_per_trigger=1,
            sdk_max_retries=0, timeout_seconds=150, actions_per_plan=[1, 4],
            included_in_original_execution_cost_tables=False),
        preedit_external_tool_types=dict(tool_names), summaries=summaries,
        verification=dict(source_logs_unchanged=True, original_matched_cells=16,
            distinct_run_paths=len(configs), all_task_denominators_complete=True,
            configuration_mismatches=problems, preedit_budget_assertions_passed=True))
    write('report.json', result)
    dump_csv('summary.csv', summaries)
    dump_csv('per_sample.csv', rows)
    print(json.dumps({k: result[k] for k in ['actual_planner_requests', 'actual_planner_response_models',
        'actual_reasoning_effort', 'actual_planner_sampling_fields', 'recorded_qwen_sampling_fields', 'verification']}, ensure_ascii=False))
    print(json.dumps([r for r in summaries if r['cohort'] == cohort], ensure_ascii=False))


if __name__ == '__main__':
    main()
