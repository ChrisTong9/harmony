"""Retry transient tool failures without changing a plan or masking exhaustion."""
from audio_edit.utils.execution_outcome import open_decoded_audio


def valid_audio(path):
    import numpy as np
    with open_decoded_audio(path) as source:
        if not len(source):
            raise ValueError('Empty audio file')
        for block in source.blocks(blocksize=65536, dtype='float32'):
            if not np.isfinite(block).all():
                raise ValueError('Non-finite audio file')

    return path


def bounded_call(ctx, name, call, *, validate=None, attempts=2):
    for attempt in range(1, attempts+1):
        audit = dict(tool=name, attempt=attempt, maximum_attempts=attempts)
        ctx.params.setdefault('tool_attempts', []).append(audit)
        try:
            result = call()
            if validate:
                validate(result)
            audit['status'] = 'completed'
            return result
        except Exception as exc:
            audit.update(status='error', error=f'{type(exc).__name__}: {exc}')

            if attempt == attempts:
                raise
