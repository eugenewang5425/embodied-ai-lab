import pytest

from monocular_depth.config import UPSTREAM_ROOT, checkpoint_for
from monocular_depth.download import verify_checkpoint
from monocular_depth.model import upstream_model_class


def test_checkpoint_variants_and_hash_gate(tmp_path):
    assert checkpoint_for("metric") != checkpoint_for("relative")
    with pytest.raises(ValueError):
        checkpoint_for("incorrect")
    fake = tmp_path / "invalid.pth"
    fake.write_bytes(b"not official weights")
    with pytest.raises(ValueError, match="SHA256"):
        verify_checkpoint(fake, "metric")


@pytest.mark.skipif(not UPSTREAM_ROOT.exists(), reason="requires pinned upstream checkout")
def test_official_relative_and_metric_imports_are_isolated():
    relative = upstream_model_class(UPSTREAM_ROOT, "relative")
    metric = upstream_model_class(UPSTREAM_ROOT, "metric")
    assert relative is not metric
    assert relative.__module__ != metric.__module__
    assert upstream_model_class(UPSTREAM_ROOT, "relative") is relative
