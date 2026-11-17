import pytest
from pydantic import ValidationError

from app.config.settings import PipelineConfig, load_config
from app.core.logging import redact


def test_defaults_are_valid_and_safe():
    cfg = PipelineConfig()
    assert cfg.project.mode == "text_to_image"
    assert cfg.deduplication.auto_delete is False and cfg.processing.preserve_originals is True
    assert cfg.export.remote_upload_enabled is False and cfg.images.auto_upscale is False


@pytest.mark.parametrize("patch", [
    {"deduplication": {"perceptual_threshold": "configurable"}},   # placeholder strings are not numbers
    {"deduplication": {"semantic_threshold": "configurable"}},
    {"deduplication": {"auto_delete": True}},
    {"processing": {"preserve_originals": False}},
    {"export": {"remote_upload_enabled": True}},
    {"images": {"auto_upscale": True}},
    {"splits": {"train": 0.8, "validation": 0.05, "test": 0.05}},  # does not sum to 1
    {"quality": {"manual_review_band": [80, 20]}},
    {"quality": {"min_overall_score": 140}},
    {"project": {"mode": "everything"}},
    {"rights": {"allowed_statuses": ["public_domain_probably"]}},
    {"not_a_section": {}},
    {"quality": {"unknown_key": 1}},
])
def test_invalid_configuration_is_rejected(patch):
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(patch)


def test_example_yaml_loads(tmp_path):
    from pathlib import Path

    example = Path(__file__).resolve().parents[3] / "configs" / "pipeline_config.example.yaml"
    assert example.exists()
    cfg = load_config(example)
    assert cfg.splits.seed == 42


def test_config_hash_ignores_machine_paths():
    a = PipelineConfig.model_validate({"paths": {"sources": ["/tmp/a"], "output_dir": "/x"}})
    b = PipelineConfig.model_validate({"paths": {"sources": ["/home/b"], "output_dir": "/y"}})
    assert a.config_hash() == b.config_hash()
    c = PipelineConfig.model_validate({"quality": {"min_overall_score": 70}})
    assert c.config_hash() != PipelineConfig().config_hash()
    assert "/tmp/a" not in str(a.public_dict())


def test_log_redaction_hides_secrets_and_home():
    from pathlib import Path

    out = redact(f"api_key=sk-12345 token: abcdef Authorization: Bearer xyz.123 path {Path.home()}/pics/a.jpg")
    assert "sk-12345" not in out and "abcdef" not in out and "xyz.123" not in out
    assert str(Path.home()) not in out and "~/pics/a.jpg" in out
