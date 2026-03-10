"""Minimal LiveKit VAD plugin wrapping Silero VAD for use with non-streaming STT."""
import os
import time
import numpy as np
import torch
from livekit import rtc
from livekit.agents import vad, utils


class SileroVADPlugin(vad.VAD):
    def __init__(self, model_dir: str, threshold: float = 0.5):
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.032))
        self._model_dir = model_dir
        self._threshold = threshold

    def stream(self, **kwargs):
        return SileroVADStream(self, self._model_dir, self._threshold)


class SileroVADStream(vad.VADStream):
    def __init__(self, vad_instance, model_dir, threshold):
        super().__init__(vad=vad_instance)
        self._model_dir = model_dir
        self._threshold = threshold
        self._model = None
        self._speaking = False
        self._silence_count = 0
        self._speech_count = 0
        self._samples_index = 0
        self._speech_start_time = 0.0
        self._pending_frames = []

    def _ensure_model(self):
        if self._model is None:
            jit_path = os.path.join(self._model_dir, "silero_vad.jit")
            if os.path.exists(jit_path):
                self._model = torch.jit.load(jit_path, map_location="cpu")
            else:
                self._model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=False,
                )
            self._model.eval()

    def _process_frame(self, frame: rtc.AudioFrame):
        self._ensure_model()
        audio_data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        self._samples_index += len(audio_data)

        for i in range(0, len(audio_data), 512):
            chunk = audio_data[i:i + 512]
            if len(chunk) < 512:
                chunk = np.pad(chunk, (0, 512 - len(chunk)))

            tensor = torch.from_numpy(chunk).float().unsqueeze(0)
            prob = self._model(tensor, 16000).item()
            now = time.time()

            if prob >= self._threshold:
                self._silence_count = 0
                self._speech_count += 1
                if not self._speaking and self._speech_count >= 2:
                    self._speaking = True
                    self._speech_start_time = now
                    self._event_ch.send_nowait(vad.VADEvent(
                        type=vad.VADEventType.START_OF_SPEECH,
                        samples_index=self._samples_index,
                        timestamp=now,
                        speech_duration=0.0,
                        silence_duration=0.0,
                        speaking=True,
                        probability=prob,
                    ))
            else:
                self._speech_count = 0
                if self._speaking:
                    self._silence_count += 1
                    if self._silence_count >= 10:
                        speech_dur = now - self._speech_start_time
                        self._speaking = False
                        self._silence_count = 0
                        self._event_ch.send_nowait(vad.VADEvent(
                            type=vad.VADEventType.END_OF_SPEECH,
                            samples_index=self._samples_index,
                            timestamp=now,
                            speech_duration=speech_dur,
                            silence_duration=0.32,
                            speaking=False,
                            probability=prob,
                            frames=self._pending_frames.copy(),
                        ))
                        self._pending_frames.clear()

            if self._speaking:
                self._pending_frames.append(frame)

            self._event_ch.send_nowait(vad.VADEvent(
                type=vad.VADEventType.INFERENCE_DONE,
                samples_index=self._samples_index,
                timestamp=now,
                speech_duration=now - self._speech_start_time if self._speaking else 0.0,
                silence_duration=self._silence_count * 0.032,
                speaking=self._speaking,
                probability=prob,
            ))

    async def _main_task(self):
        """Main loop: read frames from input channel and process them."""
        import logging
        log = logging.getLogger("voxlabs.vad_lk")
        log.info("VAD _main_task started, waiting for audio frames...")
        frame_count = 0

        async for frame in self._input_ch:
            frame_count += 1
            if frame_count == 1:
                log.info("VAD received first audio frame (type=%s)", type(frame).__name__)
            if isinstance(frame, rtc.AudioFrame):
                self._process_frame(frame)
            else:
                log.debug("VAD received non-audio frame: %s", type(frame).__name__)

        if self._speaking:
            now = time.time()
            self._speaking = False
            self._event_ch.send_nowait(vad.VADEvent(
                type=vad.VADEventType.END_OF_SPEECH,
                samples_index=self._samples_index,
                timestamp=now,
                speech_duration=now - self._speech_start_time,
                silence_duration=0.0,
                speaking=False,
                probability=0.0,
                frames=self._pending_frames.copy(),
            ))
            self._pending_frames.clear()
