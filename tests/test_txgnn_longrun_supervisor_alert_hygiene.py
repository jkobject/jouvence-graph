import importlib.util
import sys
from pathlib import Path
from subprocess import CompletedProcess

WATCHDOG_PATH = Path("/Users/jkobject/.hermes/scripts/txgnn_longrun_supervisor.py")
_spec = importlib.util.spec_from_file_location("txgnn_longrun_supervisor", WATCHDOG_PATH)
assert _spec and _spec.loader
watchdog = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = watchdog
_spec.loader.exec_module(watchdog)


def write_run(rd: Path, *, command: str, rc: str | None = None, disposition: str | None = None) -> None:
    rd.mkdir(parents=True)
    (rd / "meta.json").write_text(command)
    (rd / "stdout.log").write_text("started\n")
    if rc is not None:
        (rd / "rc").write_text(rc)
    if disposition is not None:
        (rd / watchdog.ACK_FILE).write_text(
            '{"status":"%s","accepted_by_task":"t_test","success_claim":false}' % disposition
        )


def configure_tmp_watchdog(tmp_path: Path, monkeypatch) -> Path:
    work = tmp_path / "work"
    report = work / "artifacts" / "reports" / "longrun_supervisor"
    monkeypatch.setattr(watchdog, "WORK", work)
    monkeypatch.setattr(watchdog, "REPORT", report)
    monkeypatch.setattr(watchdog, "STALE_SECONDS", 1)
    return report


def test_non_txgnn_hermes_cron_prompt_does_not_trigger_vm_guard_missing() -> None:
    ps_text = (
        "99157 1 S 00:00:12 0.0 0.1 "
        "/Users/jkobject/.hermes/hermes-agent/venv/bin/hermes chat -Q --source cron "
        "-q Tu es le superviseur Kanban du projet pert-gym ... Board: pert-gym "
        "... Workspace: /Users/jkobject/.openclaw/workspace/work/pert-gym ... "
        "copied task text mentioning TxGNN verify ungated canonical LaminDB PyG embedding"
    )

    findings = watchdog.check_local_processes(extra_ps_text=ps_text)

    assert [(f.severity, f.kind) for f in findings] == [("info", "non_txgnn_hermes_cron_ignored")]
    assert findings[0].evidence["board"] == "pert-gym"
    assert not any(f.kind == "vm_guard_missing" for f in findings)


def test_genuine_local_txgnn_heavy_command_still_triggers_vm_guard_missing() -> None:
    ps_text = (
        "12345 1 S 00:02:00 0.0 0.1 "
        "uv run python sync_parquet_edges_to_lamindb.py --kg-root gs://jouvencekb/main"
    )

    findings = watchdog.check_local_processes(extra_ps_text=ps_text)

    assert [(f.severity, f.kind) for f in findings] == [("warning", "vm_guard_missing")]


def test_pert_gym_remote_worker_command_does_not_trigger_txgnn_vm_guard_missing() -> None:
    ps_text = (
        "34696 1 S 00:18:21 0.0 0.1 "
        "gcloud compute ssh pert-gym-worker-eu --zone europe-west1-b --command "
        "cd ~/work/pert-gym && uv run python resume_gse216481_tfatlas_chunks_20260703.py "
        "--stage gs://scperturb/pert-gym/staging --embedding lamin"
    )

    findings = watchdog.check_local_processes(extra_ps_text=ps_text)

    assert [(f.severity, f.kind) for f in findings] == [("info", "pert_gym_remote_job_ignored")]
    assert findings[0].evidence["project"] == "pert-gym"
    assert findings[0].evidence["vm"] == "pert-gym-worker-eu"
    assert not any(f.kind == "vm_guard_missing" for f in findings)


def test_non_txgnn_worker_with_jouvence_signal_still_triggers_vm_guard_missing() -> None:
    ps_text = (
        "34696 1 S 00:18:21 0.0 0.1 "
        "gcloud compute ssh pert-gym-worker-eu --zone europe-west1-b --command "
        "cd ~/work/pert-gym && uv run python resume.py "
        "--stage gs://scperturb/pert-gym/staging --kg-root gs://jouvencekb/main --embedding lamin"
    )

    findings = watchdog.check_local_processes(extra_ps_text=ps_text)

    assert [(f.severity, f.kind) for f in findings] == [("warning", "vm_guard_missing")]


def test_non_txgnn_worker_with_jouvence_signal_and_incidental_required_vm_still_alerts() -> None:
    ps_text = (
        "34696 1 S 00:18:21 0.0 0.1 "
        "gcloud compute ssh pert-gym-worker-eu --zone europe-west1-b --command "
        "cd ~/work/pert-gym && uv run python resume.py "
        "--stage gs://scperturb/pert-gym/staging --kg-root gs://jouvencekb/main "
        "--embedding lamin --note txgnn-worker"
    )

    findings = watchdog.check_local_processes(extra_ps_text=ps_text)

    assert [(f.severity, f.kind) for f in findings] == [("warning", "vm_guard_missing")]


def test_txgnn_worker_with_jouvence_signal_is_visibly_guarded() -> None:
    ps_text = (
        "34696 1 S 00:18:21 0.0 0.1 "
        "gcloud compute ssh txgnn-worker --zone europe-west1-b --command "
        "cd ~/work/TxGNN && uv run python sync_parquet_edges_to_lamindb.py "
        "--kg-root gs://jouvencekb/main"
    )

    findings = watchdog.check_local_processes(extra_ps_text=ps_text)

    assert findings == []


def test_txgnn_worker_with_leading_gcloud_flags_is_visibly_guarded() -> None:
    ps_text = (
        "34696 1 S 00:18:21 0.0 0.1 "
        "gcloud compute ssh --zone europe-west1-b txgnn-worker --command "
        "cd ~/work/TxGNN && uv run python sync_parquet_edges_to_lamindb.py "
        "--kg-root gs://jouvencekb/main"
    )

    findings = watchdog.check_local_processes(extra_ps_text=ps_text)

    assert findings == []


def test_pert_gym_rxrx19b_lamin_dry_run_does_not_trigger_txgnn_vm_guard_missing() -> None:
    ps_text = (
        "80858 1 S 00:00:25 0.0 0.1 "
        "uv run python tools/register_rxrx19b_lamin_payload.py "
        "--payload-dir /Users/jkobject/.openclaw/workspace/work/pert-gym/artifacts/schema_audit/rxrx19b_recursion_dl_embedding_qa_t_32145964_20260703/local_payload "
        "--report-json artifacts/schema_audit/rxrx_required_datasets_lamin_status_20260703.dryrun_registration.json "
        "--dry-run"
    )

    findings = watchdog.check_local_processes(extra_ps_text=ps_text)

    assert [(f.severity, f.kind) for f in findings] == [("info", "pert_gym_lamin_dry_run_ignored")]
    assert findings[0].evidence["project"] == "pert-gym"
    assert findings[0].evidence["mode"] == "dry-run"
    assert not any(f.kind == "vm_guard_missing" for f in findings)


def test_pert_gym_rxrx19b_lamin_non_dry_run_without_jouvence_scope_is_ignored() -> None:
    ps_text = (
        "80858 1 S 00:00:25 0.0 0.1 "
        "uv run python tools/register_rxrx19b_lamin_payload.py "
        "--payload-dir /Users/jkobject/.openclaw/workspace/work/pert-gym/artifacts/schema_audit/rxrx19b_recursion_dl_embedding_qa_t_32145964_20260703/local_payload "
        "--report-json artifacts/schema_audit/rxrx_required_datasets_lamin_status_20260703.registration.json"
    )

    findings = watchdog.check_local_processes(extra_ps_text=ps_text)

    assert findings == []


def test_unmarked_new_unsafe_and_stale_runs_still_alert(tmp_path: Path, monkeypatch) -> None:
    report = configure_tmp_watchdog(tmp_path, monkeypatch)
    rd = report / "lamindb" / "new_unsafe_stale"
    write_run(
        rd,
        command='{"cmd":"uv run python sync_parquet_edges_to_lamindb.py --kg-root /Users/jkobject/mnt/gcs/jouvencekb/main"}',
    )

    findings = watchdog.check_run_sidecars(now=9999999999)

    severe = {(f.severity, f.kind) for f in findings}
    assert ("critical", "mac_fuse_heavy_launch_artifact") in severe
    assert ("warning", "run_missing_vm_guard") in severe
    assert ("critical", "stale_stdout_no_rc") in severe


def test_marked_unsafe_superseded_run_is_info_with_audit_trail(tmp_path: Path, monkeypatch) -> None:
    report = configure_tmp_watchdog(tmp_path, monkeypatch)
    rd = report / "lamindb" / "legacy_stale"
    write_run(
        rd,
        command='{"cmd":"uv run python sync_parquet_edges_to_lamindb.py --kg-root /Users/jkobject/mnt/gcs/jouvencekb/main"}',
        disposition="unsafe_superseded",
    )

    findings = watchdog.check_run_sidecars(now=9999999999)

    assert findings
    assert all(f.severity == "info" for f in findings)
    assert {f.kind for f in findings} == {
        "acknowledged_mac_fuse_heavy_launch_artifact",
        "acknowledged_run_missing_vm_guard",
        "acknowledged_stale_stdout_no_rc",
    }
    assert all(f.evidence["legacy_disposition"]["status"] == "unsafe_superseded" for f in findings)


def test_marked_failed_legacy_rc_is_info_not_repeated_page(tmp_path: Path, monkeypatch) -> None:
    report = configure_tmp_watchdog(tmp_path, monkeypatch)
    rd = report / "remap" / "legacy_failed"
    write_run(
        rd,
        command='{"wrapper":"/tmp/run_remap_fresh_udc.py","udc_dir":"/tmp/udc"}',
        rc="1\n",
        disposition="failed_legacy",
    )

    findings = watchdog.check_run_sidecars(now=9999999999)

    assert [(f.severity, f.kind) for f in findings] == [
        ("info", "acknowledged_run_missing_vm_guard"),
        ("info", "acknowledged_run_failed_rc"),
    ]
    assert all(f.evidence["legacy_disposition"]["status"] == "failed_legacy" for f in findings)


def test_vm_readonly_ssh_command_includes_default_zone(monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_sh(cmd, cwd=watchdog.WORK, timeout=30):
        commands.append(cmd)
        if cmd[:4] == ["gcloud", "compute", "ssh", watchdog.REQUIRED_VM]:
            return CompletedProcess(cmd, 0, f"{watchdog.REQUIRED_VM}\n", "")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(watchdog.shutil, "which", lambda name: "/usr/bin/gcloud" if name == "gcloud" else None)
    monkeypatch.setattr(watchdog, "REQUIRED_VM_ZONE", "europe-west1-b")
    monkeypatch.setattr(watchdog, "sh", fake_sh)

    findings = watchdog.check_vm_readonly()

    assert findings == []
    ssh_cmd = commands[-1]
    assert ssh_cmd[:4] == ["gcloud", "compute", "ssh", watchdog.REQUIRED_VM]
    assert "--zone" in ssh_cmd
    assert ssh_cmd[ssh_cmd.index("--zone") + 1] == "europe-west1-b"


def test_vm_readonly_ssh_command_includes_discovered_zone(monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_sh(cmd, cwd=watchdog.WORK, timeout=30):
        commands.append(cmd)
        if cmd[:4] == ["gcloud", "config", "get-value", "compute/zone"]:
            return CompletedProcess(cmd, 0, "(unset)\n", "")
        if cmd[:4] == ["gcloud", "compute", "instances", "list"]:
            return CompletedProcess(cmd, 0, "europe-west1-b\n", "")
        if cmd[:4] == ["gcloud", "compute", "ssh", watchdog.REQUIRED_VM]:
            return CompletedProcess(cmd, 0, f"{watchdog.REQUIRED_VM}\n", "")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(watchdog.shutil, "which", lambda name: "/usr/bin/gcloud" if name == "gcloud" else None)
    monkeypatch.setattr(watchdog, "REQUIRED_VM_ZONE", None)
    monkeypatch.setattr(watchdog, "sh", fake_sh)

    findings = watchdog.check_vm_readonly()

    assert findings == []
    ssh_cmd = commands[-1]
    assert ssh_cmd[:4] == ["gcloud", "compute", "ssh", watchdog.REQUIRED_VM]
    assert "--zone" in ssh_cmd
    assert ssh_cmd[ssh_cmd.index("--zone") + 1] == "europe-west1-b"


def test_vm_readonly_failed_ssh_still_surfaces_warning(monkeypatch) -> None:
    def fake_sh(cmd, cwd=watchdog.WORK, timeout=30):
        if cmd[:4] == ["gcloud", "config", "get-value", "compute/zone"]:
            return CompletedProcess(cmd, 0, "europe-west1-b\n", "")
        if cmd[:4] == ["gcloud", "compute", "ssh", watchdog.REQUIRED_VM]:
            return CompletedProcess(cmd, 255, "", "Permission denied")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(watchdog.shutil, "which", lambda name: "/usr/bin/gcloud" if name == "gcloud" else None)
    monkeypatch.setattr(watchdog, "REQUIRED_VM_ZONE", None)
    monkeypatch.setattr(watchdog, "sh", fake_sh)

    findings = watchdog.check_vm_readonly()

    assert [(f.severity, f.kind) for f in findings] == [("warning", "vm_check_failed")]
    assert findings[0].evidence["returncode"] == 255
    assert findings[0].evidence["zone"] == "europe-west1-b"
    assert "Permission denied" in findings[0].evidence["stderr"]


def test_vm_process_parser_ignores_grep_helper_for_single_real_writer() -> None:
    stdout = "\n".join([
        "txgnn-worker",
        "123 1 S 00:12:03 bigBedToBed -bed shard.bed -udcDir /tmp/udc input.bb output.bed",
        "456 455 S 00:00:00 grep -E sync_parquet_edges_to_lamindb|build_remap_motif|bigBedToBed|run_remap_fresh_udc|pyg|embedding",
        "789 788 S 00:00:00 sh -c hostname; ps axo pid=,ppid=,stat=,etime=,command= | grep -E 'sync_parquet_edges_to_lamindb|build_remap_motif|bigBedToBed|run_remap_fresh_udc|pyg|embedding' || true",
    ])

    host, writer_lines = watchdog.vm_writer_lines_from_stdout(stdout)

    assert host == "txgnn-worker"
    assert writer_lines == [
        "123 1 S 00:12:03 bigBedToBed -bed shard.bed -udcDir /tmp/udc input.bb output.bed"
    ]


def test_vm_duplicate_writer_risk_still_fires_for_two_real_writers(monkeypatch) -> None:
    def fake_sh(cmd, cwd=watchdog.WORK, timeout=30):
        if cmd[:4] == ["gcloud", "config", "get-value", "compute/zone"]:
            return CompletedProcess(cmd, 0, "europe-west1-b\n", "")
        if cmd[:4] == ["gcloud", "compute", "ssh", watchdog.REQUIRED_VM]:
            stdout = "\n".join([
                "txgnn-worker",
                "123 1 S 00:12:03 bigBedToBed -bed shard.bed -udcDir /tmp/udc input.bb output.bed",
                "124 1 S 00:10:00 uv run python sync_parquet_edges_to_lamindb.py --kg-root gs://jouvencekb/main",
                "456 455 S 00:00:00 grep -E sync_parquet_edges_to_lamindb|build_remap_motif|bigBedToBed|run_remap_fresh_udc|pyg|embedding",
            ])
            return CompletedProcess(cmd, 0, stdout, "")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(watchdog.shutil, "which", lambda name: "/usr/bin/gcloud" if name == "gcloud" else None)
    monkeypatch.setattr(watchdog, "REQUIRED_VM_ZONE", None)
    monkeypatch.setattr(watchdog, "sh", fake_sh)

    findings = watchdog.check_vm_readonly()

    assert [(f.severity, f.kind) for f in findings] == [("critical", "vm_duplicate_writer_risk")]
    assert len(findings[0].evidence["processes"]) == 2
    assert all("grep -E" not in proc for proc in findings[0].evidence["processes"])
