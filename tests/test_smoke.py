from __future__ import annotations

import grain_morph


def test_version_present():
    assert isinstance(grain_morph.__version__, str)
    assert grain_morph.__version__
