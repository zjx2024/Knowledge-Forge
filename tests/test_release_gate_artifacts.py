import inspect

from knowledge_forge.service import KnowledgeForgeService


def test_current_gate_accepts_caller_metrics_characterization():
    signature = inspect.signature(KnowledgeForgeService.evaluate_policy_snapshot_gate)
    assert "control_metrics" in signature.parameters
    assert "treatment_metrics" in signature.parameters
