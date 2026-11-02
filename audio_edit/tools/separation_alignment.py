"""Remove only the SAM codec's documented right-frame padding, never retime."""


def target_alignment(input_samples, input_rate, output_samples, output_rate, hop_length):
    values = (input_samples, input_rate, output_samples, output_rate, hop_length)
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError("Source alignment needs positive integer sample geometry")
    expected = (input_samples * output_rate + input_rate - 1) // input_rate
    frame_padded = ((expected + hop_length - 1) // hop_length) * hop_length
    if output_samples not in (expected, frame_padded):
        raise ValueError(
            "Separated source length is neither the original length nor its codec-frame padding"
        )
    return dict(
        code="source_target_alignment",
        original_samples=input_samples,
        original_rate=input_rate,
        model_samples=output_samples,
        output_samples=expected,
        output_rate=output_rate,
        codec_hop_length=hop_length,
        removed_right_padding=output_samples - expected,
        policy="Keep sample zero and every original-time sample; trim codec right padding only, without resampling or stretching",
    )
