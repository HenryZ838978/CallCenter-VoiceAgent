"""DTLN (Dual-Signal Transformation LSTM Network) real-time speech denoiser.

Two-stage ONNX model: Stage 1 (freq domain mask) + Stage 2 (time domain enhancement).
Processes 512-sample blocks (32ms at 16kHz) with ~8ms latency on CPU.

Based on: https://github.com/breizhn/DTLN
"""
import os
import logging
import numpy as np
import onnxruntime as ort

log = logging.getLogger("voxlabs.denoiser")

BLOCK_LEN = 512
BLOCK_SHIFT = 128
FFT_LEN = 512


class DTLNDenoiser:
    def __init__(self, model_dir: str):
        self._model_dir = model_dir
        self._model1 = None
        self._model2 = None
        self._state1 = None
        self._state2 = None
        self._in_buffer = None
        self._out_buffer = None

    def load(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        m1_path = os.path.join(self._model_dir, "model_1.onnx")
        m2_path = os.path.join(self._model_dir, "model_2.onnx")
        self._model1 = ort.InferenceSession(m1_path, opts, providers=["CPUExecutionProvider"])
        self._model2 = ort.InferenceSession(m2_path, opts, providers=["CPUExecutionProvider"])
        self.reset()
        log.info("DTLNDenoiser loaded (model_dir=%s)", self._model_dir)
        return self

    def reset(self):
        self._state1 = np.zeros((1, 2, 128, 2), dtype=np.float32)
        self._state2 = np.zeros((1, 2, 128, 2), dtype=np.float32)
        self._in_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)
        self._out_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)

    def process_chunk(self, audio: np.ndarray) -> np.ndarray:
        """Process a single 512-sample block. Returns denoised 512-sample block."""
        if len(audio) != BLOCK_LEN:
            if len(audio) < BLOCK_LEN:
                audio = np.pad(audio, (0, BLOCK_LEN - len(audio)))
            else:
                audio = audio[:BLOCK_LEN]

        self._in_buffer = audio.astype(np.float32)

        in_mag = np.abs(np.fft.rfft(self._in_buffer)).reshape(1, 1, -1).astype(np.float32)
        in_phase = np.angle(np.fft.rfft(self._in_buffer))

        out1, self._state1 = self._model1.run(
            None,
            {"input_2": in_mag, "input_3": self._state1},
        )

        estimated_mag = out1[0]
        estimated_complex = estimated_mag * np.exp(1j * in_phase)
        estimated_block = np.real(np.fft.irfft(estimated_complex.squeeze())).astype(np.float32)

        in_time = estimated_block.reshape(1, 1, -1).astype(np.float32)
        out2, self._state2 = self._model2.run(
            None,
            {"input_4": in_time, "input_5": self._state2},
        )

        return out2[0].squeeze()

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process arbitrary-length audio. Returns denoised audio of same length."""
        original_len = len(audio)
        audio = audio.astype(np.float32)

        pad_len = (BLOCK_LEN - len(audio) % BLOCK_LEN) % BLOCK_LEN
        if pad_len > 0:
            audio = np.pad(audio, (0, pad_len))

        self.reset()
        output = np.zeros_like(audio)
        for i in range(0, len(audio), BLOCK_LEN):
            block = audio[i:i + BLOCK_LEN]
            output[i:i + BLOCK_LEN] = self.process_chunk(block)

        return output[:original_len]
