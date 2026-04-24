from pathlib import Path
from safetensors.torch import load_file
import torch

from kame.models import lm
from kame.models.loaders import _upgrade_legacy_lm_state_dict
from kame.utils.utils import cross_entropy


def _get_assets() -> Path:
    return Path(__file__).parent / "assets"


def _get_lm(device=None, dtype=torch.float32) -> lm.LMModel:
    torch.manual_seed(1234)
    model = lm.LMModel(
        delays=[0, 1, 2, 4],
        n_q=3,
        dep_q=3,
        card=32,
        text_card=48,
        dim=16,
        num_layers=2,
        num_heads=1,
        hidden_scale=1,
        depformer_dim=16,
        depformer_multi_linear=True,
        depformer_weights_per_step=True,
        depformer_weights_per_step_schedule=[0, 1, 1],
        depformer_low_rank_embeddings=8,
        depformer_num_heads=1,
        depformer_gating="silu",
        context=4,
        device=device,
        dtype=dtype,
    )
    return model


def _get_legacy_lm_state_dict() -> dict[str, torch.Tensor]:
    return load_file(_get_assets() / "test_lm_model.safetensors")


def test_init():
    _get_lm(dtype=torch.float32)
    _get_lm(dtype=torch.bfloat16)
    _get_lm(dtype=torch.float16)


def test_upgrade_legacy_lm_state_dict():
    legacy_state = _get_legacy_lm_state_dict()
    assert not any(key.startswith("oracle_emb.") for key in legacy_state)

    upgraded_state = _upgrade_legacy_lm_state_dict(legacy_state)

    expected_oracle_keys = {
        f"oracle_emb.{key[len('text_emb.') :]}" for key in legacy_state if key.startswith("text_emb.")
    }
    actual_oracle_keys = {key for key in upgraded_state if key.startswith("oracle_emb.")}
    assert actual_oracle_keys == expected_oracle_keys

    for oracle_key in actual_oracle_keys:
        text_key = f"text_emb.{oracle_key[len('oracle_emb.') :]}"
        assert torch.equal(upgraded_state[oracle_key], legacy_state[text_key])
        assert upgraded_state[oracle_key].data_ptr() != legacy_state[text_key].data_ptr()


def test_upgrade_legacy_lm_state_dict_is_noop_for_current_checkpoints():
    state = _get_lm().state_dict()
    upgraded_state = _upgrade_legacy_lm_state_dict(state)

    assert upgraded_state.keys() == state.keys()
    for key, value in state.items():
        assert torch.equal(upgraded_state[key], value)


@torch.no_grad
def test_forward():
    model = _get_lm()
    state = _upgrade_legacy_lm_state_dict(_get_legacy_lm_state_dict())
    model.load_state_dict(state)
    codes = load_file(_get_assets() / "test_lm_codes.safetensors")["codes"]
    out = model(codes)
    assert out.logits is not None
    assert out.text_logits is not None
    assert out.mask.shape == codes[:, 1:].shape
    assert out.text_mask.shape == codes[:, :1].shape
    assert out.logits.shape[:-1] == codes[:, 1:].shape
    assert out.logits.shape[-1] == model.card
    assert out.text_logits.shape[-1] == model.text_card

    ref_out = load_file(_get_assets() / "test_lm_out.safetensors")
    assert (ref_out["mask"] == out.mask).all()
    assert (ref_out["text_mask"] == out.text_mask).all()
    ce = cross_entropy(out.logits, codes[:, 1:], out.mask)
    ce_ref = cross_entropy(ref_out["logits"], codes[:, 1:], out.mask)
    delta = (ce.mean(dim=(0, 2)) - ce_ref.mean(dim=(0, 2))).abs() / ce_ref.mean(dim=(0, 2))
    assert delta.amax() <= 1e-6, delta.amax()

    ce = cross_entropy(out.text_logits, codes[:, :1], out.text_mask)
    ce_ref = cross_entropy(ref_out["text_logits"], codes[:, :1], out.text_mask)
    delta = (ce.mean(dim=(0, 2)) - ce_ref.mean(dim=(0, 2))).abs() / ce_ref.mean(dim=(0, 2))
    assert delta.amax() <= 1e-6, delta.amax()
