from __future__ import annotations

import itertools
import json
import random
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app import coverage
from app.main import create_app


# ---------------------------------------------------------------------------
# 纯算法测试
# ---------------------------------------------------------------------------


def brute_force(roster, frames):
    """枚举所有排法，作为 DP 的对照答案（只用于小样本测试）。"""
    first = frames[0].frame_no
    last = frames[-1].frame_no
    scanned = {f.frame_no: f for f in frames}
    span = last - first + 1

    results = []
    for assignment in itertools.product(range(len(roster)), repeat=span):
        ok = True
        for pos in range(span):
            page = assignment[pos]
            frame = scanned.get(first + pos)
            if frame is not None and frame.candidates:
                if roster[page] not in frame.candidates:
                    ok = False
                    break
            if pos:
                delta = page - assignment[pos - 1]
                if delta not in (0, 1):
                    ok = False
                    break
        if ok:
            results.append(assignment)

    pages = {}
    for i, plate in enumerate(roster):
        scanned_possible = set()
        missed_possible = set()
        guaranteed = bool(results)
        for assignment in results:
            covered = False
            for pos, page in enumerate(assignment):
                if page != i:
                    continue
                frame_no = first + pos
                if frame_no in scanned:
                    scanned_possible.add(frame_no)
                    covered = True
                else:
                    missed_possible.add(frame_no)
            if not covered:
                guaranteed = False
        pages[plate] = {
            "possible_scanned_frames": sorted(scanned_possible),
            "possible_missed_frames": sorted(missed_possible),
            "guaranteed": guaranteed,
        }
    return bool(results), pages


def make_frames(spec):
    return [
        coverage.Frame(frame_no=no, candidates=tuple(cands))
        for no, cands in sorted(spec.items())
    ]


def test_matches_brute_force_on_random_small_cases():
    rng = random.Random(20261122)
    for _ in range(120):
        roster = [str(i + 1) for i in range(rng.randint(2, 4))]
        span = rng.randint(1, 5)
        frames = []
        for pos in range(span):
            if rng.random() < 0.35:
                continue
            k = rng.randint(0, 2)
            cands = tuple(rng.sample(roster, k)) if k < len(roster) else tuple(roster)
            frames.append(coverage.Frame(frame_no=100 + pos, candidates=cands))
        if not frames:
            continue

        expected_ok, expected_pages = brute_force(roster, frames)
        try:
            result = coverage.compute_volume(roster, frames)
            actual_ok = True
        except coverage.ArrangementError:
            actual_ok = False

        assert actual_ok == expected_ok
        if not expected_ok:
            continue
        for plate in roster:
            got = result["pages"][plate]
            want = expected_pages[plate]
            assert got["possible_scanned_frames"] == want["possible_scanned_frames"], (
                roster, frames, plate
            )
            assert got["possible_missed_frames"] == want["possible_missed_frames"], (
                roster, frames, plate
            )
            assert got["guaranteed"] == want["guaranteed"], (roster, frames, plate)


def test_guaranteed_vs_pending_vs_missing():
    roster = ["1", "2", "3"]
    # 帧 1 认不出（版1或2），帧 2=版2。排法只有 (1,2) 和 (2,2)。
    frames = make_frames({1: (), 2: ("2",)})
    result = coverage.compute_volume(roster, frames)
    pages = result["pages"]

    assert pages["1"]["guaranteed"] is False
    assert pages["1"]["possible_scanned_frames"] == [1]
    assert pages["2"]["guaranteed"] is True
    assert pages["2"]["possible_scanned_frames"] == [1, 2]
    assert pages["3"]["possible_scanned_frames"] == []


def test_missing_page_points_to_missed_frames_in_span():
    roster = ["1", "2", "3", "4", "5"]
    # 帧 1=版1、帧 3=版3，帧 2 是首末已扫帧之间的跳号。
    frames = make_frames({1: ("1",), 3: ("3",)})
    result = coverage.compute_volume(roster, frames)
    pages = result["pages"]

    assert pages["1"]["guaranteed"] is True
    assert pages["2"]["possible_scanned_frames"] == []
    assert pages["2"]["possible_missed_frames"] == [2]
    assert pages["3"]["guaranteed"] is True
    # 漏扫帧 2 夹在版1和版3之间，够不到版4、5。
    assert pages["4"]["possible_missed_frames"] == []
    assert pages["5"]["possible_missed_frames"] == []


def test_infeasible_volume_raises():
    roster = ["1", "2", "3"]
    frames = make_frames({1: ("3",), 2: ("1",)})
    with pytest.raises(coverage.ArrangementError):
        coverage.compute_volume(roster, frames)


def test_span_limit():
    roster = [str(i + 1) for i in range(600)]
    frames = make_frames({1: (), 501: ()})
    with pytest.raises(coverage.ArrangementError) as info:
        coverage.compute_volume(roster, frames)
    assert info.value.code == "span_too_large"

    frames = make_frames({1: (), 500: ()})
    assert coverage.compute_volume(roster, frames)["span"]["frame_count"] == 500


def test_large_500_frame_volume_is_fast():
    roster = [str(i + 1) for i in range(500)]
    frames = [coverage.Frame(frame_no=i + 1, candidates=()) for i in range(500)]
    frames[0] = coverage.Frame(frame_no=1, candidates=("1",))
    frames[-1] = coverage.Frame(frame_no=500, candidates=("500",))
    # 500 帧、500 版，首末钉死，逐帧只能前进 1，于是每帧恰好对一版。
    result = coverage.compute_volume(roster, frames)
    assert result["span"]["frame_count"] == 500
    for plate in ("1", "2", "250", "500"):
        assert result["pages"][plate]["guaranteed"] is True
        assert result["pages"][plate]["possible_scanned_frames"] == [int(plate)]


# ---------------------------------------------------------------------------
# API 测试
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    app = create_app(str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def submit(client, issue, code, frames, plates=None):
    body = {"volume_code": code, "frames": frames}
    if plates is not None:
        body["plates"] = plates
    return client.post(f"/issues/{issue}/volumes", json=body)


def test_submit_and_check_flow(client):
    plates = ["1", "2", "3", "4"]
    r = submit(
        client,
        "2026-11-22",
        "A",
        [
            {"frame_no": 1, "candidates": ["2"]},
            {"frame_no": 2, "candidates": []},
        ],
        plates,
    )
    assert r.status_code == 201, r.text
    assert r.json()["issue_created"] is True

    # 帧1=版2；帧2 认不出，可以是版2或版3。
    # 版1、4 不可能由本卷已扫帧拍到 -> 缺扫且无漏扫帧线索。
    report = client.get("/issues/2026-11-22/plates").json()
    status = {p["plate"]: p["status"] for p in report["plates"]}
    assert status == {"1": "missing", "2": "scanned", "3": "pending", "4": "missing"}
    assert report["summary"] == {"plate_count": 4, "scanned": 1, "pending": 1, "missing": 2}

    missing4 = next(p for p in report["plates"] if p["plate"] == "4")
    assert missing4["possible_volumes_and_missed_frames"] == []
    assert missing4["hint"]


def test_pending_plate_lists_candidates(client):
    plates = ["1", "2", "3"]
    submit(client, "I", "A", [
        {"frame_no": 10, "candidates": []},
        {"frame_no": 11, "candidates": ["2", "3"]},
    ], plates)
    report = client.get("/issues/I/plates").json()
    by = {p["plate"]: p for p in report["plates"]}

    # 可行排法：12、22、23、33。版1 只在 12 中出现，版2 在 33 中没有，
    # 版3 只在 23、33 中出现。
    assert by["1"]["status"] == "pending"
    assert by["1"]["possible_volumes_and_frames"] == [
        {"volume_code": "A", "frame_no": 10},
    ]
    assert by["2"]["status"] == "pending"
    assert by["2"]["possible_volumes_and_frames"] == [
        {"volume_code": "A", "frame_no": 10},
        {"volume_code": "A", "frame_no": 11},
    ]
    assert by["3"]["status"] == "pending"
    assert by["3"]["possible_volumes_and_frames"] == [
        {"volume_code": "A", "frame_no": 10},
        {"volume_code": "A", "frame_no": 11},
    ]


def test_pending_from_multiple_volumes(client):
    submit(client, "I", "A", [
        {"frame_no": 1, "candidates": ["1", "2"]},
        {"frame_no": 2, "candidates": ["2"]},
    ], ["1", "2", "3"])
    submit(client, "I", "B", [
        {"frame_no": 5, "candidates": ["1", "2"]},
    ])
    report = client.get("/issues/I/plates").json()
    by = {p["plate"]: p for p in report["plates"]}
    # 版1 在两卷里都只在部分排法下被拍到，线索要合并列出。
    assert by["1"]["status"] == "pending"
    assert by["1"]["possible_volumes_and_frames"] == [
        {"volume_code": "A", "frame_no": 1},
        {"volume_code": "B", "frame_no": 5},
    ]
    assert by["2"]["status"] == "scanned"


def test_missing_lists_volume_and_missed_frame(client):
    plates = ["1", "2", "3", "4", "5"]
    submit(client, "I", "A", [
        {"frame_no": 1, "candidates": ["1"]},
        {"frame_no": 3, "candidates": ["3"]},
    ], plates)
    report = client.get("/issues/I/plates").json()
    by = {p["plate"]: p for p in report["plates"]}
    # 唯一排法 123：漏扫帧 2 只能落在版2。
    assert by["2"]["status"] == "missing"
    assert by["2"]["possible_volumes_and_missed_frames"] == [
        {"volume_code": "A", "frame_no": 2}
    ]
    assert by["2"]["hint"] is None
    # 版4、5 连漏扫帧也够不着 -> 提示另找胶片。
    for plate in ("4", "5"):
        assert by[plate]["status"] == "missing"
        assert by[plate]["possible_volumes_and_missed_frames"] == []
        assert by[plate]["hint"]


def test_new_volume_only_recomputes_itself(client, tmp_path):
    plates = ["1", "2", "3"]
    submit(client, "I", "A", [{"frame_no": 1, "candidates": ["1"]}], plates)
    before = client.get("/issues/I/volumes/A").json()

    r = submit(client, "I", "B", [{"frame_no": 5, "candidates": []}])
    assert r.status_code == 201, r.text
    assert r.json()["roster_changed"] is False
    after = client.get("/issues/I/volumes/A").json()
    assert after["updated_at"] == before["updated_at"]


def test_roster_change_recomputes_all_volumes(client):
    submit(client, "I", "A", [{"frame_no": 1, "candidates": ["1"]}], ["1", "2"])
    r = submit(client, "I", "B", [{"frame_no": 3, "candidates": ["3"]}], ["1", "2", "3"])
    assert r.status_code == 201, r.text
    assert r.json()["roster_changed"] is True
    assert r.json()["recomputed_volume_codes"] == ["A"]
    a = client.get("/issues/I/volumes/A").json()
    assert set(a["pages"]) == {"1", "2", "3"}


def test_infeasible_roster_change_rolls_back_everything(client):
    submit(client, "I", "A", [{"frame_no": 1, "candidates": ["3"]}], ["1", "2", "3"])
    before_issue = client.get("/issues/I").json()
    before_a = client.get("/issues/I/volumes/A").json()

    # 新清单只有 1、2，卷 A 的帧候选 3 失效 -> 整体回滚
    r = submit(client, "I", "B", [{"frame_no": 5, "candidates": ["1"]}], ["1", "2"])
    assert r.status_code == 422
    assert r.json()["error"]["data_unchanged"] is True

    assert client.get("/issues/I").json()["plates"] == before_issue["plates"]
    assert client.get("/issues/I/volumes/A").json()["frames"] == before_a["frames"]
    assert client.get("/issues/I/volumes/B").status_code == 404


def test_infeasible_new_issue_is_not_created(client):
    r = submit(client, "J", "A", [{"frame_no": 2, "candidates": ["3"]},
                                  {"frame_no": 1, "candidates": ["1"]}],
               ["1", "2", "3"])
    assert r.status_code == 422
    assert client.get("/issues/J").status_code == 404


def test_resubmitting_same_volume_uses_new_frames(client):
    submit(client, "I", "A", [{"frame_no": 1, "candidates": ["1"]}], ["1", "2"])
    r = submit(client, "I", "A", [{"frame_no": 1, "candidates": ["2"]}])
    assert r.status_code == 201
    a = client.get("/issues/I/volumes/A").json()
    assert a["frames"] == [{"frame_no": 1, "candidates": ["2"]}]


def test_invalid_inputs(client):
    base = [{"frame_no": 1, "candidates": ["1"]}]
    assert submit(client, "I", "A", base).status_code == 422  # 首期缺清单
    assert submit(client, "I", "A", base, []).status_code == 422
    assert submit(client, "I", "A", base, ["1", "1"]).status_code == 422
    assert submit(client, "I", "A", [{"frame_no": 1, "candidates": ["9"]}], ["1"]).status_code == 422
    assert submit(client, "I", "A", [{"frame_no": 0, "candidates": ["1"]}], ["1"]).status_code == 422
    assert submit(client, "I", "A", [
        {"frame_no": 1, "candidates": ["1"]},
        {"frame_no": 1, "candidates": ["1"]},
    ], ["1"]).status_code == 422
    assert submit(client, "I", "A", [{"frame_no": 1, "candidates": ["1"]}], ["1"]).status_code == 201


def test_missing_issue_and_plate_404(client):
    submit(client, "I", "A", [{"frame_no": 1, "candidates": ["1"]}], ["1", "2"])
    assert client.get("/issues/NOPE/plates").status_code == 404
    assert client.get("/issues/I/plates/9").status_code == 404
    assert client.get("/issues/I/volumes/Z").status_code == 404


def test_db_persists_across_app_restart(tmp_path):
    db = str(tmp_path / "persist.db")
    with TestClient(create_app(db)) as c:
        assert submit(c, "I", "A", [{"frame_no": 1, "candidates": ["1"]}], ["1"]).status_code == 201
    with TestClient(create_app(db)) as c:
        assert c.get("/issues/I/volumes/A").status_code == 200
        assert c.get("/issues/I/plates").json()["summary"]["scanned"] == 1


# ---------------------------------------------------------------------------
# 存储失败回滚
# ---------------------------------------------------------------------------


def test_write_failure_rolls_back_and_keeps_old_data(client, monkeypatch):
    import sqlite3

    from app import db as db_module

    submit(client, "I", "A", [{"frame_no": 1, "candidates": ["1"]}], ["1", "2"])

    calls = {"n": 0}
    real = db_module._frames_to_json

    def flaky(frames):
        calls["n"] += 1
        if calls["n"] >= 1:
            raise sqlite3.OperationalError("disk I/O error")
        return real(frames)

    monkeypatch.setattr(db_module, "_frames_to_json", flaky)

    r = submit(client, "I", "B", [{"frame_no": 3, "candidates": ["2"]}], ["1", "2", "3"])
    assert r.status_code == 503
    assert r.json()["error"]["data_unchanged"] is True

    monkeypatch.setattr(db_module, "_frames_to_json", real)
    # 清单、已有卷都保持提交前状态，新卷没有落库。
    assert client.get("/issues/I").json()["plates"] == ["1", "2"]
    assert set(client.get("/issues/I/volumes/A").json()["pages"]) == {"1", "2"}
    assert client.get("/issues/I/volumes/B").status_code == 404


def test_blank_or_non_string_plate_rejected(client):
    body = {
        "volume_code": "A",
        "plates": ["1", "  "],
        "frames": [{"frame_no": 1, "candidates": []}],
    }
    r = client.post("/issues/I/volumes", json=body)
    assert r.status_code == 422
    assert r.json()["error"]["code"] in {"invalid_plate", "empty_roster"}
    assert client.get("/issues/I").status_code == 404


def test_plate_order_is_preserved(client):
    plates = ["A", "B", "C"]
    submit(client, "I", "A", [{"frame_no": 1, "candidates": ["A"]}], plates)
    report = client.get("/issues/I/plates").json()
    assert [p["plate"] for p in report["plates"]] == plates
    assert client.get("/issues/I").json()["plates"] == plates


def test_set_roster_endpoint_recomputes_and_rolls_back(client):
    submit(client, "I", "A", [{"frame_no": 1, "candidates": ["1"]}], ["1", "2"])

    ok = client.put("/issues/I/plates", json={"plates": ["1", "2", "3"]})
    assert ok.status_code == 200, ok.text
    assert ok.json()["recomputed_volume_codes"] == ["A"]
    assert set(client.get("/issues/I/volumes/A").json()["pages"]) == {"1", "2", "3"}

    # 新清单去掉卷 A 用到的候选版 1 -> 排不通，全部保留旧状态。
    bad = client.put("/issues/I/plates", json={"plates": ["2", "3"]})
    assert bad.status_code == 422
    assert bad.json()["error"]["data_unchanged"] is True
    assert client.get("/issues/I").json()["plates"] == ["1", "2", "3"]
    assert set(client.get("/issues/I/volumes/A").json()["pages"]) == {"1", "2", "3"}


# ---------------------------------------------------------------------------
# 单版线索索引
# ---------------------------------------------------------------------------


def _db_path(client) -> str:
    return client.app.state.store.db_path


def _index_rows(db_path, issue, plate):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT volume_code, guaranteed, scanned_json, missed_json "
            "FROM plate_index WHERE issue_id = ? AND plate = ? ORDER BY volume_code",
            (issue, plate),
        ).fetchall()
    finally:
        conn.close()


def _index_meta(db_path, issue):
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT plates_json FROM index_meta WHERE issue_id = ?", (issue,)
        ).fetchone()
        return None if row is None else json.loads(row[0])
    finally:
        conn.close()


def _seed_two_volumes(client, issue="I", plates=("1", "2", "3", "4", "5")):
    # 卷 A：帧1=版1、帧3=版3 -> 版2 缺扫，漏扫帧 2 是唯一线索；版1/3 已扫。
    submit(client, issue, "A", [
        {"frame_no": 1, "candidates": ["1"]},
        {"frame_no": 3, "candidates": ["3"]},
    ], list(plates))
    # 卷 B：帧5 认不出 -> 版2 在部分排法下可被已扫帧 5 拍到 -> 待核。
    submit(client, issue, "B", [{"frame_no": 5, "candidates": []}])


def test_plate_endpoint_matches_full_report_for_every_plate(client):
    plates = ["1", "2", "3", "4", "5"]
    _seed_two_volumes(client)

    full = client.get("/issues/I/plates").json()
    for item in full["plates"]:
        single = client.get(f"/issues/I/plates/{item['plate']}")
        assert single.status_code == 200, single.text
        assert single.json() == {"issue_id": "I", **item}


def test_index_is_built_on_write_and_only_reads_target_plate_rows(client):
    _seed_two_volumes(client)
    db_path = _db_path(client)

    # 新库写入即建好索引，meta 版序快照与清单一致。
    assert _index_meta(db_path, "I") == ["1", "2", "3", "4", "5"]
    rows = _index_rows(db_path, "I", "2")
    assert [r[0] for r in rows] == ["A", "B"]
    # 卷 A 的版2 无已扫线索；卷 B 的版2 有已扫帧 5。
    assert json.loads(rows[0][2]) == []
    assert json.loads(rows[1][2]) == [5]

    # 单版查询不反序列化任何卷的完整 result_json。
    from app import db as db_module

    parsed = []
    real_loads = db_module.json.loads

    def tracking_loads(value):
        parsed.append(value)
        return real_loads(value)

    db_module.json.loads = tracking_loads
    try:
        r = client.get("/issues/I/plates/2")
    finally:
        db_module.json.loads = real_loads

    assert r.status_code == 200
    assert r.json()["status"] == "pending"
    conn = sqlite3.connect(db_path)
    try:
        result_payloads = [
            row[0] for row in conn.execute("SELECT result_json FROM volumes").fetchall()
        ]
    finally:
        conn.close()
    assert not any(payload in parsed for payload in result_payloads)
    # 有已扫线索时连漏扫帧清单都不解析：卷 A 的 missed_json 不出现在解析入参里。
    missed_a = next(r[3] for r in rows if r[0] == "A")
    assert missed_a not in parsed


def test_scanned_hint_skipped_but_missing_plate_still_lists_missed_frames(client):
    # 只提交卷 A：帧1=版1、帧3=版3，版2 缺扫但漏扫帧 2 可达，版5 够不着。
    submit(client, "I", "A", [
        {"frame_no": 1, "candidates": ["1"]},
        {"frame_no": 3, "candidates": ["3"]},
    ], ["1", "2", "3", "4", "5"])

    # 版2 没有任何已扫线索：仍给出漏扫帧线索，且与整期报告完全一致。
    missing = client.get("/issues/I/plates/2").json()
    full = next(
        p for p in client.get("/issues/I/plates").json()["plates"] if p["plate"] == "2"
    )
    assert missing == {"issue_id": "I", **full}
    assert missing["status"] == "missing"
    assert missing["possible_volumes_and_missed_frames"] == [
        {"volume_code": "A", "frame_no": 2}
    ]
    assert missing["hint"] is None

    # 版5 连漏扫帧也够不着 -> hint 提示另找胶片。
    far = client.get("/issues/I/plates/5").json()
    assert far["status"] == "missing"
    assert far["possible_volumes_and_missed_frames"] == []
    assert far["hint"]


def test_resubmit_volume_replaces_only_its_index_rows(client):
    _seed_two_volumes(client)
    db_path = _db_path(client)

    before_b = _index_rows(db_path, "I", "2")
    # 换交卷 A：改成帧1=版1 一帧，版2 不再有卷 A 的漏扫线索。
    submit(client, "I", "A", [{"frame_no": 1, "candidates": ["1"]}])

    after_a = _index_rows(db_path, "I", "2")
    b_rows = [r for r in after_a if r[0] == "B"]
    assert len(b_rows) == 1
    # 版序未变，卷 B 的索引行原封不动。
    assert b_rows == [r for r in before_b if r[0] == "B"]
    a_rows = [r for r in after_a if r[0] == "A"]
    assert len(a_rows) == 1
    assert json.loads(a_rows[0][3]) == []  # 卷 A 已无漏扫帧

    # 单版结论与整期报告仍一致。
    full = client.get("/issues/I/plates").json()
    for item in full["plates"]:
        assert client.get(f"/issues/I/plates/{item['plate']}").json() == {
            "issue_id": "I",
            **item,
        }


def test_removing_last_scanned_clue_reports_still_valid_missed_frames(client):
    _seed_two_volumes(client)
    # 初始：版2 靠卷 B 帧5 成为待核（有已扫线索）。
    assert client.get("/issues/I/plates/2").json()["status"] == "pending"

    # 换交卷 B，撤掉这条已扫线索：帧7 钉在版4，版2 再无已扫帧拍到。
    submit(client, "I", "B", [{"frame_no": 7, "candidates": ["4"]}])

    report = client.get("/issues/I/plates/2").json()
    assert report["status"] == "missing"
    full = next(
        p for p in client.get("/issues/I/plates").json()["plates"] if p["plate"] == "2"
    )
    assert report == {"issue_id": "I", **full}
    # 卷 A 的漏扫帧 2 仍是有效线索。
    assert {"volume_code": "A", "frame_no": 2} in report[
        "possible_volumes_and_missed_frames"
    ]
    assert report["hint"] is None


def test_failed_resubmit_keeps_old_index_and_old_answer(client):
    _seed_two_volumes(client)
    before = client.get("/issues/I/plates/2").json()
    db_path = _db_path(client)
    conn = sqlite3.connect(db_path)
    try:
        before_rows = conn.execute(
            "SELECT volume_code, scanned_json, missed_json FROM plate_index "
            "WHERE issue_id = 'I' ORDER BY volume_code, plate"
        ).fetchall()
    finally:
        conn.close()

    # 候选版 9 不在清单 -> 422，整次提交（含索引）回滚。
    bad = submit(client, "I", "B", [{"frame_no": 7, "candidates": ["9"]}])
    assert bad.status_code == 422
    assert bad.json()["error"]["data_unchanged"] is True

    assert client.get("/issues/I/plates/2").json() == before
    conn = sqlite3.connect(db_path)
    try:
        after_rows = conn.execute(
            "SELECT volume_code, scanned_json, missed_json FROM plate_index "
            "WHERE issue_id = 'I' ORDER BY volume_code, plate"
        ).fetchall()
    finally:
        conn.close()
    assert after_rows == before_rows


def test_roster_change_rebuilds_index_with_rejudged_volumes(client):
    _seed_two_volumes(client, plates=("1", "2", "3"))
    db_path = _db_path(client)

    r = client.put("/issues/I/plates", json={"plates": ["1", "2", "3", "4"]})
    assert r.status_code == 200, r.text
    assert _index_meta(db_path, "I") == ["1", "2", "3", "4"]

    conn = sqlite3.connect(db_path)
    try:
        indexed_plates = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT plate FROM plate_index WHERE issue_id = 'I'"
            ).fetchall()
        }
    finally:
        conn.close()
    assert indexed_plates == {"1", "2", "3", "4"}

    full = client.get("/issues/I/plates").json()
    for item in full["plates"]:
        assert client.get(f"/issues/I/plates/{item['plate']}").json() == {
            "issue_id": "I",
            **item,
        }


def test_legacy_db_backfills_index_on_first_plate_query_and_reuses(client):
    _seed_two_volumes(client)
    db_path = _db_path(client)

    # 模拟旧库：删掉索引表数据与 meta，只留 issues/volumes 里的已存结果。
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM plate_index")
        conn.execute("DELETE FROM index_meta")
        conn.commit()
    finally:
        conn.close()
    assert _index_meta(db_path, "I") is None

    # 首次单版查询：用已存结果补建，结论与整期报告一致。
    first = client.get("/issues/I/plates/2")
    assert first.status_code == 200, first.text
    full = next(
        p for p in client.get("/issues/I/plates").json()["plates"] if p["plate"] == "2"
    )
    assert first.json() == {"issue_id": "I", **full}

    assert _index_meta(db_path, "I") == ["1", "2", "3", "4", "5"]
    rows_after_first = _index_rows(db_path, "I", "2")
    assert rows_after_first

    # 第二次查询直接复用，不再补建：把补建函数替换成会报错的哨兵。
    from app import db as db_module

    def boom(*_args, **_kwargs):
        raise AssertionError("索引已存在，不应再次补建")

    original = db_module.Store._backfill_index
    db_module.Store._backfill_index = boom
    try:
        again = client.get("/issues/I/plates/2")
    finally:
        db_module.Store._backfill_index = original
    assert again.json() == first.json()
    assert _index_rows(db_path, "I", "2") == rows_after_first


def test_backfill_retries_after_busy_and_succeeds(client, monkeypatch):
    _seed_two_volumes(client)
    db_path = _db_path(client)

    # 旧库：无索引。
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM plate_index")
        conn.execute("DELETE FROM index_meta")
        conn.commit()
    finally:
        conn.close()

    from app import db as db_module

    real_backfill = db_module.Store._backfill_index
    calls = {"n": 0}

    def flaky_backfill(self, conn, issue_id, roster, now):
        calls["n"] += 1
        if calls["n"] == 1:
            # 模拟补建与换交撞写锁：首次补建拿到 BUSY，触发回滚重试。
            raise sqlite3.OperationalError("database is locked")
        return real_backfill(self, conn, issue_id, roster, now)

    monkeypatch.setattr(db_module.Store, "_backfill_index", flaky_backfill)

    r = client.get("/issues/I/plates/2")
    assert r.status_code == 200, r.text
    assert calls["n"] == 2  # 首次撞锁回滚，重试成功
    monkeypatch.undo()

    # 重试后索引可用且与整期报告一致。
    full = next(
        p for p in client.get("/issues/I/plates").json()["plates"] if p["plate"] == "2"
    )
    assert r.json() == {"issue_id": "I", **full}


def test_backfill_after_concurrent_resubmit_serves_new_clues(client):
    """旧库补建窗口内换交已提交：补建后的查询必须反映换交后的新线索。"""

    _seed_two_volumes(client)
    db_path = _db_path(client)

    # 旧库无索引。
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM plate_index")
        conn.execute("DELETE FROM index_meta")
        conn.commit()
    finally:
        conn.close()

    from app import db as db_module

    # 换交先完成（撤掉卷 B 对版2 的已扫线索），随后首次单版查询触发补建，
    # 补建读取的是换交后的卷结果。
    store = db_module.Store(db_path)
    store.submit_volume("I", "B", None, [{"frame_no": 7, "candidates": ["4"]}])

    report = client.get("/issues/I/plates/2").json()
    full = next(
        p for p in client.get("/issues/I/plates").json()["plates"] if p["plate"] == "2"
    )
    assert report == {"issue_id": "I", **full}
    assert report["status"] == "missing"
    assert {"volume_code": "A", "frame_no": 2} in report[
        "possible_volumes_and_missed_frames"
    ]


def test_plate_index_endpoint_404s(client):
    _seed_two_volumes(client)
    assert client.get("/issues/NOPE/plates/1").status_code == 404
    assert client.get("/issues/I/plates/9").status_code == 404
    # 404 不应在旧库上留下半成品索引。
    assert _index_meta(_db_path(client), "NOPE") is None


def test_roster_first_then_volume_maintains_index(client):
    # 先单独登记清单（此时无卷，只写 meta），再交卷应增量建本卷索引。
    r = client.put("/issues/R/plates", json={"plates": ["1", "2", "3"]})
    assert r.status_code == 200
    assert _index_meta(_db_path(client), "R") == ["1", "2", "3"]
    assert _index_rows(_db_path(client), "R", "1") == []

    submit(client, "R", "A", [{"frame_no": 1, "candidates": ["1"]}])
    rows = _index_rows(_db_path(client), "R", "1")
    assert len(rows) == 1 and rows[0][0] == "A"
    full = client.get("/issues/R/plates").json()
    assert client.get("/issues/R/plates/1").json() == {
        "issue_id": "R",
        **next(p for p in full["plates"] if p["plate"] == "1"),
    }


def test_guaranteed_merge_across_volumes_via_index(client):
    # 两卷都单独保证版1 已扫：guaranteed_by_volumes 合并两卷。
    submit(client, "I", "A", [{"frame_no": 1, "candidates": ["1"]}], ["1", "2"])
    submit(client, "I", "B", [{"frame_no": 2, "candidates": ["1"]}])
    report = client.get("/issues/I/plates/1").json()
    full = next(p for p in client.get("/issues/I/plates").json()["plates"]
                if p["plate"] == "1")
    assert report == {"issue_id": "I", **full}
    assert report["status"] == "scanned"
    assert report["guaranteed_by_volumes"] == ["A", "B"]
    assert report["scanned_frames"] == [
        {"volume_code": "A", "frame_no": 1},
        {"volume_code": "B", "frame_no": 2},
    ]


def test_real_concurrent_backfill_and_resubmit(tmp_path):
    """补建事务与换交事务真实撞锁：查询最终必须给出换交后的有效线索。"""

    import threading

    from app import db as db_module

    db_path = str(tmp_path / "conc.db")
    setup = db_module.Store(db_path)
    setup.initialize()
    setup.submit_volume(
        "I",
        "A",
        ["1", "2", "3", "4", "5"],
        [
            {"frame_no": 1, "candidates": ["1"]},
            {"frame_no": 3, "candidates": ["3"]},
        ],
    )
    setup.submit_volume("I", "B", None, [{"frame_no": 5, "candidates": []}])

    # 退化为旧库：索引为空。
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM plate_index")
        conn.execute("DELETE FROM index_meta")
        conn.commit()
    finally:
        conn.close()

    errors: list[Exception] = []

    def resubmit() -> None:
        try:
            db_module.Store(db_path).submit_volume(
                "I", "B", None, [{"frame_no": 7, "candidates": ["4"]}]
            )
        except Exception as exc:  # pragma: no cover - 仅用于暴露并发问题
            errors.append(exc)

    def query() -> None:
        try:
            db_module.Store(db_path).get_plate_clues("I", "2")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    t1 = threading.Thread(target=resubmit)
    t2 = threading.Thread(target=query)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not errors

    # 换交已生效，索引随之可用；版2 缺扫，卷 A 漏扫帧 2 仍是有效线索。
    report = client_get_single(db_path, "I", "2")
    assert report["status"] == "missing"
    assert {"volume_code": "A", "frame_no": 2} in report[
        "possible_volumes_and_missed_frames"
    ]
    assert _index_meta(db_path, "I") == ["1", "2", "3", "4", "5"]


def client_get_single(db_path, issue, plate):
    from app import checking
    from app import db as db_module

    clues = db_module.Store(db_path).get_plate_clues(issue, plate)
    return {"issue_id": issue, **checking.build_single_plate_report(clues)}
