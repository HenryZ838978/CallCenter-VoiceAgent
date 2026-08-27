"""Speaker-aware VAD — combines Silero VAD with ECAPA-TDNN speaker verification.

Inspired by FireRedChat pVAD (https://huggingface.co/FireRedTeam/FireRedChat-pvad).
Instead of filtering barge-in by VAD threshold alone, this module verifies that
the detected speech belongs to the enrolled primary speaker, not TTS echo or bystanders.

Usage:
    svad = SpeakerAwareVAD(vad_model_dir, speaker_model_dir).load()
    # First user utterance → enroll speaker
    svad.enroll_speaker(audio_np_float32)
    # Subsequent chunks → speaker-verified VAD
    result = svad.process_chunk(chunk)
    # result["is_target_speaker"] is True only if speech matches enrolled speaker
"""
import os
import logging
import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("voxlabs.svad")

SPEAKER_EMBED_DIM = 192
ENROLL_MIN_SAMPLES = 8000       # 0.5s minimum for enrollment
VERIFY_WINDOW_SAMPLES = 16000   # 1s window for speaker verification
SIMILARITY_THRESHOLD = 0.25     # cosine similarity threshold for speaker match


class SpeakerAwareVAD:
    """Silero VAD + ECAPA-TDNN speaker embedding for personalized barge-in detection."""

    def __init__(self, vad_model_dir: str, speaker_model_dir: str,
                 threshold: float = 0.5, speaker_threshold: float = SIMILARITY_THRESHOLD):
        self.threshold = threshold
        self.speaker_threshold = speaker_threshold
        self._vad_model_dir = vad_model_dir
        self._speaker_model_dir = speaker_model_dir

        self._vad_model = None
        self._speaker_encoder = None
        self._target_embedding = None
        self._enrolled = False

        self._speech_active = False
        self._silence_count = 0
        self._recent_speech_buffer = []
        self._recent_speech_samples = 0

    def load(self, device: str = "cpu"):
        self._device = device

        jit_path = os.path.join(self._vad_model_dir, "silero_vad.jit")
        if os.path.exists(jit_path):
            self._vad_model = torch.jit.load(jit_path, map_location="cpu")
        else:
            self._vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=False,
            )
        self._vad_model.eval()

        import torchaudio
        if not hasattr(torchaudio, 'list_audio_backends'):
            torchaudio.list_audio_backends = lambda: ['soundfile']

        if not os.environ.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        from speechbrain.inference.speaker import EncoderClassifier
        self._speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=self._speaker_model_dir,
            run_opts={"device": device},
        )

        self.reset()
        log.info("SpeakerAwareVAD loaded (Silero + ECAPA-TDNN, device=%s)", device)
        return self

    def reset(self):
        if self._vad_model is not None:
            self._vad_model.reset_states()
        self._speech_active = False
        self._silence_count = 0
        self._recent_speech_buffer = []
        self._recent_speech_samples = 0
        self._verify_counter = 0
        self._last_similarity = 0.0
        self._last_is_target = False

    def enroll_speaker(self, audio: np.ndarray, sr: int = 16000):
        """Enroll the primary speaker from a speech segment.
        Call after the first complete user utterance for best results.
        """
        if len(audio) < ENROLL_MIN_SAMPLES:
            log.warning("Enrollment audio too short (%d samples, need %d)", len(audio), ENROLL_MIN_SAMPLES)
            return False

        trim = min(len(audio), sr * 10)
        audio_trimmed = audio[:trim]

        tensor = torch.from_numpy(audio_trimmed).float().unsqueeze(0).to(self._device)
        with torch.no_grad():
            embedding = self._speaker_encoder.encode_batch(tensor)
        self._target_embedding = F.normalize(embedding.squeeze(), dim=0)
        self._enrolled = True
        log.info("Speaker enrolled (%.1fs audio, embedding norm=%.3f)",
                 len(audio_trimmed) / sr, self._target_embedding.norm().item())
        return True

    def verify_speaker(self, audio: np.ndarray) -> float:
        """Check if audio matches enrolled speaker. Returns cosine similarity [-1, 1]."""
        if not self._enrolled or self._target_embedding is None:
            return 1.0

        if len(audio) < 3200:
            return 0.5

        tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self._device)
        with torch.no_grad():
            embedding = self._speaker_encoder.encode_batch(tensor)
        query_emb = F.normalize(embedding.squeeze(), dim=0)
        similarity = F.cosine_similarity(self._target_embedding, query_emb, dim=0).item()
        return similarity

    def process_chunk(self, audio_chunk: np.ndarray, sr: int = 16000) -> dict:
        """Process a 32ms chunk. Returns standard VAD dict + speaker verification fields.

        Extra fields:
            is_target_speaker: bool — True if speech matches enrolled speaker
            speaker_similarity: float — cosine similarity (only computed when speech detected)
        """
        tensor = torch.from_numpy(audio_chunk).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        prob = self._vad_model(tensor, sr).item()

        event = {
            "speech_prob": prob,
            "speech_start": False,
            "speech_end": False,
            "is_target_speaker": False,
            "speaker_similarity": 0.0,
        }

        if prob >= self.threshold:
            self._recent_speech_buffer.append(audio_chunk.copy())
            self._recent_speech_samples += len(audio_chunk)
            while self._recent_speech_samples > VERIFY_WINDOW_SAMPLES:
                removed = self._recent_speech_buffer.pop(0)
                self._recent_speech_samples -= len(removed)
        else:
            if not self._speech_active:
                self._recent_speech_buffer = []
                self._recent_speech_samples = 0

        if prob >= self.threshold and not self._speech_active:
            self._speech_active = True
            self._silence_count = 0
            event["speech_start"] = True
        elif prob < self.threshold and self._speech_active:
            self._silence_count += 1
            if self._silence_count >= 15:
                self._speech_active = False
                self._silence_count = 0
                event["speech_end"] = True
        elif prob >= self.threshold and self._speech_active:
            self._silence_count = 0

        if prob >= self.threshold and self._enrolled and self._recent_speech_samples >= 3200:
            self._verify_counter += 1
            if self._verify_counter >= 3:
                recent_audio = np.concatenate(self._recent_speech_buffer)
                similarity = self.verify_speaker(recent_audio)
                self._last_similarity = similarity
                self._last_is_target = similarity >= self.speaker_threshold
                self._verify_counter = 0
            event["speaker_similarity"] = self._last_similarity
            event["is_target_speaker"] = self._last_is_target

        elif prob >= self.threshold and not self._enrolled:
            event["is_target_speaker"] = True
            event["speaker_similarity"] = 1.0

        return event

    @property
    def is_speech_active(self):
        return self._speech_active

    @property
    def is_enrolled(self):
        return self._enrolled
