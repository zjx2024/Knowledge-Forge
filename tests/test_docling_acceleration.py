import sys
from types import SimpleNamespace

import pytest

from knowledge_forge import chunking
from knowledge_forge.config import Config


def test_docling_auto_selects_cuda_when_memory_is_safe(monkeypatch):
    monkeypatch.setattr(Config, "docling_device", "auto")
    monkeypatch.setattr(Config, "docling_cuda_fallback", True)
    monkeypatch.setattr(chunking, "_docling_cuda_memory", lambda: (True, 18000, 48000, ""))

    device, metadata = chunking._select_docling_device()

    assert device == "cuda:0"
    assert metadata["docling_gpu_free_mb_before"] == 18000


def test_docling_auto_uses_cpu_below_memory_guard(monkeypatch):
    monkeypatch.setattr(Config, "docling_device", "auto")
    monkeypatch.setattr(Config, "docling_cuda_fallback", True)
    monkeypatch.setattr(
        chunking,
        "_docling_cuda_memory",
        lambda: (False, 3000, 24000, "剩余显存不足"),
    )

    device, metadata = chunking._select_docling_device()

    assert device == "cpu"
    assert metadata["docling_gpu_fallback_reason"] == "剩余显存不足"


def test_forced_cuda_can_disable_cpu_fallback(monkeypatch):
    monkeypatch.setattr(Config, "docling_device", "cuda")
    monkeypatch.setattr(Config, "docling_cuda_fallback", False)
    monkeypatch.setattr(
        chunking,
        "_docling_cuda_memory",
        lambda: (False, 0, 0, "CUDA 不可用"),
    )

    with pytest.raises(RuntimeError, match="CUDA 不可用"):
        chunking._select_docling_device()


def test_docling_cuda_oom_retries_on_cpu(monkeypatch, tmp_path):
    source = tmp_path / "scan.pdf"
    source.write_bytes(b"not-used-by-mock")
    calls = []
    released = []

    class FakeCuda:
        @staticmethod
        def set_per_process_memory_fraction(fraction, device):
            assert fraction == 0.2
            assert device == 0

    def convert(_path, device):
        calls.append(device)
        if device.startswith("cuda"):
            raise RuntimeError("CUDA out of memory")
        return "# 扫描文档\n\nCPU 回退成功。"

    monkeypatch.setattr(Config, "docling_gpu_memory_fraction", 0.2)
    monkeypatch.setattr(Config, "docling_cuda_fallback", True)
    monkeypatch.setattr(chunking, "docling_available", lambda: True)
    monkeypatch.setattr(chunking, "_select_docling_device", lambda: ("cuda:0", {}))
    monkeypatch.setattr(chunking, "_docling_markdown", convert)
    monkeypatch.setattr(chunking, "_release_docling_cuda", lambda: released.append(True))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=FakeCuda()))

    parsed = chunking._parse_docling(source)

    assert calls == ["cuda:0", "cpu"]
    assert released == [True]
    assert parsed.quality["docling_accelerator"] == "cpu"
    assert "CUDA 解析失败" in parsed.quality["docling_gpu_fallback_reason"]


def test_cuda_failure_detection_is_narrow():
    assert chunking._is_cuda_failure(RuntimeError("CUDA out of memory"))
    assert not chunking._is_cuda_failure(ValueError("invalid PDF syntax"))
