"""Minimum end-to-end example: synthesize Japanese speech with Irodori-TTS-v2
using Sway Sampling (training-free inference-time schedule, F5-TTS style).

Usage:
    python examples/sway_sampling.py path/to/reference.wav "合成したいテキスト"

Recommended parameters (num_steps=6, t_schedule_mode="sway", sway_coeff=-1.0)
are chosen by sweeping CER / speaker similarity over candidate settings and
picking the best operating point. Details:
    https://magazine.kizuna-intelligence.com/articles/article-d9ac7ce68a98
"""

from __future__ import annotations

import sys
import time

import soundfile as sf
from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest


def main() -> None:
    ref_wav = sys.argv[1]
    text = "今日はいい天気ですね。"
    checkpoint_path = hf_hub_download(
        repo_id="Aratako/Irodori-TTS-500M-v2",
        filename="model.safetensors",
    )

    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=checkpoint_path,
            model_device="mps",
            # model_precision="bf16",
            codec_device="mps",
            # codec_precision="bf16",
        ),
    )

    a = time.time()
    out = runtime.synthesize(
        SamplingRequest(
            text=text,
            ref_wav=ref_wav,
            # seconds=4.0,
            num_steps=6,
            t_schedule_mode="sway",
            sway_coeff=-1.0,
            seed=42,
        ),
    )
    audio = out.audio[0].squeeze().float().cpu().numpy()
    sf.write("output.wav", audio, runtime.codec.sample_rate)
    print(
        f"saved: output.wav ({len(audio) / runtime.codec.sample_rate:.2f}s)({time.time() - a:.2f}s)",
    )


if __name__ == "__main__":
    main()
