from pathlib import Path


def test_current_repository_has_no_isolated_e2e_runner_characterization():
    assert not Path("scripts/run_isolated_e2e.py").exists()
