"""Verify prepare_inputs_labels_for_multimodal handles audios=None.

Upstream dereferences audio_features unconditionally, so a text-only or
image-only forward pass (audios=None) raised
    TypeError: 'NoneType' object is not subscriptable
and every caller had to pass a dummy torch.zeros(400, 80) waveform. This
checks both that the None path now works and that the existing audio paths
are unchanged.

Encoders are stubbed, so this runs on CPU in seconds without the 7B weights:
    PYTHONPATH=./ python tools/test_audio_optional.py
"""
import torch
import torch.nn as nn

from vita.constants import AUDIO_TOKEN_INDEX, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from vita.model.vita_arch import VITAMetaForCausalLM

DIM = 16
IMG_TOKENS = 4
AUD_TOKENS = 3


class StubVisionTower(nn.Module):
    def forward(self, images):
        return torch.ones(images.shape[0], IMG_TOKENS, DIM)


class StubAudioEncoder(nn.Module):
    def forward(self, audios, lengths):
        n = audios.shape[0]
        return {"inputs_embeds": torch.ones(n, AUD_TOKENS, DIM) * 2}


class StubConfig:
    tokenizer_model_max_length = None
    tokenizer_padding_side = "right"
    audio_prompt_num = None


class StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = StubVisionTower()
        self.audio_encoder = StubAudioEncoder()
        self.mm_projector = nn.Identity()
        self.embed = nn.Embedding(100, DIM)

    def get_vision_tower(self):
        return self.vision_tower

    def get_audio_encoder(self):
        return self.audio_encoder

    def embed_tokens(self, ids):
        return self.embed(ids)


class StubVITA(VITAMetaForCausalLM, nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self._model = StubModel()
        self.config = StubConfig()

    def get_model(self):
        return self._model

    @property
    def device(self):
        return torch.device("cpu")


def run(name, input_ids, images, audios):
    m = StubVITA()
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    out = m.prepare_inputs_labels_for_multimodal(
        input_ids, None, attention_mask, None, labels, images, audios, None
    )
    embeds = out[4]
    print(f"{name:34} -> inputs_embeds {tuple(embeds.shape)}")
    return embeds


print("=== the case that used to crash ===")
# image + text, audios=None. Before the fix this raised
# TypeError: 'NoneType' object is not subscriptable
ids = torch.tensor([[5, IMAGE_TOKEN_INDEX, 7]])
e = run("image + text, audios=None", ids, torch.ones(1, 1, 3, 8, 8), None)
assert e.shape == (1, 2 + IMG_TOKENS, DIM), e.shape

print("\n=== text only, audios=None ===")
ids = torch.tensor([[5, 6, 7]])
e = run("text only, audios=None", ids, torch.ones(1, 1, 3, 8, 8), None)
assert e.shape == (1, 3, DIM), e.shape

print("\n=== regression: existing audio paths must be unchanged ===")
aud = {
    "audios": torch.ones(1, 400, 80),
    "lengths": torch.tensor([400]),
    "lengths_for_llm": torch.tensor([AUD_TOKENS]),
}
ids = torch.tensor([[5, IMAGE_TOKEN_INDEX, AUDIO_TOKEN_INDEX, 7]])
e = run("image + audio", ids, torch.ones(1, 1, 3, 8, 8), aud)
assert e.shape == (1, 2 + IMG_TOKENS + AUD_TOKENS, DIM), e.shape

# image + dummy audio: the shape the current data pipeline always produces
ids = torch.tensor([[5, IMAGE_TOKEN_INDEX, 7]])
e = run("image + dummy audio (old behaviour)", ids, torch.ones(1, 1, 3, 8, 8), aud)
assert e.shape == (1, 2 + IMG_TOKENS, DIM), e.shape

print("\n=== <audio> token with audios=None must fail loudly ===")
ids = torch.tensor([[5, AUDIO_TOKEN_INDEX, 7]])
try:
    run("audio token but audios=None", ids, torch.ones(1, 1, 3, 8, 8), None)
    raise SystemExit("FAIL: expected an AssertionError")
except AssertionError as exc:
    print(f"correctly raised: {exc}")

print("\nALL CHECKS PASSED")
