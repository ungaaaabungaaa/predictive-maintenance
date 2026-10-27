import json

from predictive_maintenance import cli


def test_cli_download_and_run_messages_without_auth_or_hidden_fetch(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "download_fd001", lambda path: {"file": "digest"})
    assert cli.main(["download", "--data-dir", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"file": "digest"}
    calls = []

    def experiment(data_dir, output, config):
        calls.append((data_dir, output, config))
        return {"dataset_status": "verified-nasa-fd001",
                "selection": {"selected_model": "ridge"},
                "test_metrics_uncapped": {"rmse_cycles": 20.}}

    monkeypatch.setattr(cli, "run_experiment", experiment)
    assert cli.main(["run", "--data-dir", str(tmp_path), "--output", str(tmp_path / "out")]) == 0
    assert "20.000 cycles" in capsys.readouterr().out
    assert calls[0][0] == tmp_path
    assert calls[0][2].seed == 42


def test_cli_controlled_failure_and_option_validation(monkeypatch, capsys):
    def fail(path):
        raise ValueError("checksum mismatch")

    monkeypatch.setattr(cli, "download_fd001", fail)
    assert cli.main(["download"]) == 1
    assert capsys.readouterr().err == "Error: checksum mismatch\n"
    assert cli.main(["run", "--trees", "0"]) == 1
    assert "trees must be" in capsys.readouterr().err
