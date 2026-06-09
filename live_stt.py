

import os
import re
import queue
import sys
import numpy as np
import sounddevice as sd
import nemo.collections.asr as nemo_asr
from dotenv import load_dotenv
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

load_dotenv()

SAMPLE_RATE   = 16_000
LANGUAGE      = os.getenv("STT_LANGUAGE", "en-US")
CHUNK_SECONDS = 1.0

def clean(text: str) -> str:
    """Remove language tags like <en-US> from transcript."""
    return re.sub(r"<[^>]+>", "", text).strip()

def main():
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b").eval()
    model.set_inference_prompt(LANGUAGE)
    model.encoder.set_default_att_context_size([56, 3])

    cfg = model.encoder.streaming_cfg
    ch, t, ch_len = model.encoder.get_initial_cache_state(batch_size=1)
    hyps, step = None, 0
    prev_text = ""

    audio_q = queue.Queue()
    accumulated = []

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    print(f"Listening [{LANGUAGE}] — Ctrl-C to stop\n")
    print("-" * 50)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=1600,
                        callback=callback):
        try:
            while True:
                chunk = audio_q.get()
                accumulated.append(chunk)

                total_samples = sum(len(c) for c in accumulated)
                if total_samples < int(SAMPLE_RATE * CHUNK_SECONDS):
                    continue

                audio = np.concatenate(accumulated)
                accumulated = []

                buf = CacheAwareStreamingAudioBuffer(model, online_normalization=False)
                buf.append_audio(audio, stream_id=-1)

                for chunk_data, chunk_len in buf:
                    _, _, ch, t, ch_len, hyps = model.conformer_stream_step(
                        processed_signal=chunk_data,
                        processed_signal_length=chunk_len,
                        cache_last_channel=ch,
                        cache_last_time=t,
                        cache_last_channel_len=ch_len,
                        previous_hypotheses=hyps,
                        drop_extra_pre_encoded=cfg.drop_extra_pre_encoded if step else 0,
                        keep_all_outputs=buf.is_buffer_empty(),
                        return_transcription=True,
                    )
                    step += 1
                    if hyps and hyps[0].text:
                        text = clean(hyps[0].text)
                        if text and text != prev_text:
                            print(f"\r{text:<80}", end="", flush=True)
                            prev_text = text

        except KeyboardInterrupt:
            print(f"\n\nFinal transcript:\n{prev_text}\n")

if __name__ == "__main__":
    main()