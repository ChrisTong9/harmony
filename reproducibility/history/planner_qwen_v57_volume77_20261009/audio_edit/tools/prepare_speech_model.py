"""Download a pinned public Whisper model; no inference or environment changes."""
import argparse
import hashlib
import json
from pathlib import Path


MODEL_ID = 'openai/whisper-large-v3-turbo'
REVISION = '41f01f3fe87f28c78e2fbf8b568835947dd65ed9'
FILES = ['config.json', 'generation_config.json', 'model.safetensors',
         'preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json',
         'special_tokens_map.json', 'added_tokens.json', 'normalizer.json',
         'merges.txt', 'vocab.json']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(MODEL_ID, revision=REVISION, token=False, files_metadata=True)
    if info.gated or info.sha != REVISION:
        raise RuntimeError('Expected an ungated model at the pinned revision')
    sizes = {item.rfilename: item.size for item in info.siblings}
    root = Path(snapshot_download(MODEL_ID, revision=REVISION, token=False,
        local_dir=args.output, allow_patterns=FILES, max_workers=4))
    files = {}
    for name in FILES:
        path = root / name
        if not path.is_file() or path.stat().st_size != sizes[name]:
            raise RuntimeError(f'Incomplete model download: {name}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1048576), b''):
                digest.update(block)
        files[name] = dict(bytes=path.stat().st_size, sha256=digest.hexdigest())
    config = json.loads((root / 'generation_config.json').read_text())
    if not config.get('alignment_heads'):
        raise RuntimeError('Model has no alignment heads for word timestamps')
    manifest = dict(model_id=MODEL_ID, revision=REVISION, files=files,
        timestamps='Whisper cross-attention dynamic time warping; approximate',
        preparation='CPU download and checksum only; no inference validation',
        source_url=f'https://huggingface.co/{MODEL_ID}/tree/{REVISION}')
    (root / 'speech_model_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(model_id=MODEL_ID, revision=REVISION,
        files=len(files), bytes=sum(item['bytes'] for item in files.values()),
        manifest=str(root / 'speech_model_manifest.json'))))


if __name__ == '__main__':
    main()
