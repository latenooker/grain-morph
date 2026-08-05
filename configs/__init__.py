"""Data-only package exposing ``default.yaml`` to ``importlib.resources``.

Mapped to ``grain_morph.configs`` via ``[tool.setuptools] package-dir`` in
``pyproject.toml`` so the top-level ``configs/`` directory (sibling to
``src/``) ships inside the installed distribution and is loadable as a
package resource, e.g.::

    importlib.resources.files("grain_morph.configs") / "default.yaml"
"""

from __future__ import annotations
