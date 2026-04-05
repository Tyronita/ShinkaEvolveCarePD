"""
Tests for shinka_visualize Web UI endpoints.

Usage (against a running shinka_visualize instance):
    python -m pytest care_pd_task/tests/test_webui.py -v
    python -m pytest care_pd_task/tests/test_webui.py -v --base-url http://localhost:8080

On RunPod (SSH tunnel first):
    ssh -L 8080:localhost:8080 root@<pod-ip> -p <port> -i ~/.ssh/key
    python -m pytest care_pd_task/tests/test_webui.py -v
"""

import json
import pytest
import requests


def pytest_addoption(parser):
    parser.addoption("--base-url", default="http://localhost:8080",
                     help="Base URL for shinka_visualize (default: http://localhost:8080)")


@pytest.fixture(scope="session")
def base_url(request):
    return request.config.getoption("--base-url").rstrip("/")


@pytest.fixture(scope="session")
def session():
    s = requests.Session()
    s.timeout = 10
    return s


# ── Connectivity ──────────────────────────────────────────────────────────────

class TestConnectivity:
    def test_root_returns_200(self, base_url, session):
        r = session.get(f"{base_url}/")
        assert r.status_code == 200

    def test_root_is_html(self, base_url, session):
        r = session.get(f"{base_url}/")
        assert "text/html" in r.headers.get("content-type", "")

    def test_index_html_contains_shinka(self, base_url, session):
        r = session.get(f"{base_url}/index.html")
        assert r.status_code == 200
        assert len(r.text) > 100  # not empty

    def test_viz_tree_html_returns_200(self, base_url, session):
        r = session.get(f"{base_url}/viz_tree.html")
        assert r.status_code == 200

    def test_compare_html_returns_200(self, base_url, session):
        r = session.get(f"{base_url}/compare.html")
        assert r.status_code == 200


# ── JSON API ─────────────────────────────────────────────────────────────────

class TestListDatabases:
    def test_returns_200(self, base_url, session):
        r = session.get(f"{base_url}/list_databases")
        assert r.status_code == 200

    def test_returns_json_array(self, base_url, session):
        r = session.get(f"{base_url}/list_databases")
        data = r.json()
        assert isinstance(data, list)

    def test_cors_header_present(self, base_url, session):
        r = session.get(f"{base_url}/list_databases")
        assert r.headers.get("Access-Control-Allow-Origin") == "*"

    def test_db_entries_have_required_fields(self, base_url, session):
        r = session.get(f"{base_url}/list_databases")
        dbs = r.json()
        for db in dbs:
            assert "path" in db or "actual_path" in db, f"DB entry missing path: {db}"


class TestDatabaseStats:
    @pytest.fixture(scope="class")
    def first_db_path(self, base_url, session):
        r = session.get(f"{base_url}/list_databases")
        dbs = r.json()
        if not dbs:
            pytest.skip("No databases found — run at least one ShinkaEvolve generation first")
        return dbs[0].get("actual_path") or dbs[0].get("path")

    def test_stats_returns_200(self, base_url, session, first_db_path):
        r = session.get(f"{base_url}/get_database_stats", params={"db_path": first_db_path})
        assert r.status_code == 200

    def test_stats_has_program_count(self, base_url, session, first_db_path):
        r = session.get(f"{base_url}/get_database_stats", params={"db_path": first_db_path})
        data = r.json()
        assert "program_count" in data
        assert isinstance(data["program_count"], int)

    def test_program_count_returns_200(self, base_url, session, first_db_path):
        r = session.get(f"{base_url}/get_program_count", params={"db_path": first_db_path})
        assert r.status_code == 200

    def test_program_count_is_int(self, base_url, session, first_db_path):
        r = session.get(f"{base_url}/get_program_count", params={"db_path": first_db_path})
        data = r.json()
        assert "count" in data
        assert isinstance(data["count"], int)
        assert data["count"] >= 0


class TestPrograms:
    @pytest.fixture(scope="class")
    def first_db_path(self, base_url, session):
        r = session.get(f"{base_url}/list_databases")
        dbs = r.json()
        if not dbs:
            pytest.skip("No databases found")
        return dbs[0].get("actual_path") or dbs[0].get("path")

    def test_get_programs_summary_returns_200(self, base_url, session, first_db_path):
        r = session.get(f"{base_url}/get_programs_summary", params={"db_path": first_db_path})
        assert r.status_code == 200

    def test_get_programs_summary_is_list(self, base_url, session, first_db_path):
        r = session.get(f"{base_url}/get_programs_summary", params={"db_path": first_db_path})
        data = r.json()
        assert isinstance(data, list)

    def test_program_summary_fields(self, base_url, session, first_db_path):
        r = session.get(f"{base_url}/get_programs_summary", params={"db_path": first_db_path})
        programs = r.json()
        if not programs:
            pytest.skip("No programs yet")
        prog = programs[0]
        # Every program should have an id
        assert "id" in prog or "program_id" in prog, f"Missing id in program: {prog.keys()}"

    def test_get_system_prompts_returns_200(self, base_url, session, first_db_path):
        r = session.get(f"{base_url}/get_system_prompts", params={"db_path": first_db_path})
        assert r.status_code == 200

    def test_get_system_prompts_is_list(self, base_url, session, first_db_path):
        r = session.get(f"{base_url}/get_system_prompts", params={"db_path": first_db_path})
        assert isinstance(r.json(), list)


class TestMetaFiles:
    @pytest.fixture(scope="class")
    def first_db_path(self, base_url, session):
        r = session.get(f"{base_url}/list_databases")
        dbs = r.json()
        if not dbs:
            pytest.skip("No databases found")
        return dbs[0].get("actual_path") or dbs[0].get("path")

    def test_get_meta_files_returns_200(self, base_url, session, first_db_path):
        r = session.get(f"{base_url}/get_meta_files", params={"db_path": first_db_path})
        assert r.status_code == 200

    def test_get_meta_files_is_list(self, base_url, session, first_db_path):
        r = session.get(f"{base_url}/get_meta_files", params={"db_path": first_db_path})
        assert isinstance(r.json(), list)


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_missing_db_path_returns_error(self, base_url, session):
        r = session.get(f"{base_url}/get_database_stats")
        # Should return an error, not a 500 crash
        assert r.status_code in (200, 400, 404)

    def test_invalid_db_path_does_not_crash(self, base_url, session):
        r = session.get(f"{base_url}/get_programs_summary",
                        params={"db_path": "/nonexistent/path.db"})
        assert r.status_code in (200, 400, 404)
        # Should return JSON, not an HTML error page
        try:
            r.json()
        except json.JSONDecodeError:
            pytest.fail("Server returned non-JSON response for invalid db_path")

    def test_404_for_unknown_route(self, base_url, session):
        r = session.get(f"{base_url}/this_route_does_not_exist_xyz")
        assert r.status_code == 404
