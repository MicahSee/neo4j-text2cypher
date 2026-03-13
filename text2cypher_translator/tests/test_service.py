from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from text2cypher_translator.service import app, get_db, get_translator


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.extract_schema.return_value = "(:Actor {name: STRING})-[:ACTED_IN]->(:Movie {title: STRING})"
    db.run.return_value = [{"title": "Forrest Gump"}, {"title": "Cast Away"}]
    return db


@pytest.fixture
def mock_translator():
    translator = MagicMock()
    translator.generate.return_value = "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) RETURN m.title"
    return translator


@pytest.fixture
def client(mock_db, mock_translator):
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_translator] = lambda: mock_translator
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


class TestNlQuery:
    def test_returns_cypher_and_results(self, client, mock_translator, mock_db):
        resp = client.post("/api/nlquery", json={"question": "What movies did Tom Hanks act in?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["question"] == "What movies did Tom Hanks act in?"
        assert "MATCH" in body["cypher"]
        assert len(body["results"]) == 2

        mock_db.extract_schema.assert_called_once()
        mock_translator.generate.assert_called_once()

    def test_cypher_execution_failure_returns_400(self, client, mock_db):
        mock_db.run.side_effect = Exception("Invalid query")

        resp = client.post("/api/nlquery", json={"question": "bad question"})

        assert resp.status_code == 400
        assert "Cypher execution failed" in resp.json()["detail"]


class TestCypher:
    def test_run_raw_cypher(self, client, mock_db):
        resp = client.post("/neo4j/cypher", json={"cypher": "MATCH (n) RETURN n LIMIT 1"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["cypher"] == "MATCH (n) RETURN n LIMIT 1"
        assert len(body["results"]) == 2

    def test_run_cypher_with_params(self, client, mock_db):
        resp = client.post(
            "/neo4j/cypher",
            json={"cypher": "MATCH (a:Actor {name: $name}) RETURN a", "params": {"name": "Tom Hanks"}},
        )

        assert resp.status_code == 200
        mock_db.run.assert_called_with("MATCH (a:Actor {name: $name}) RETURN a", name="Tom Hanks")

    def test_cypher_error_returns_400(self, client, mock_db):
        mock_db.run.side_effect = Exception("Syntax error")

        resp = client.post("/neo4j/cypher", json={"cypher": "INVALID"})

        assert resp.status_code == 400


class TestSchema:
    def test_get_schema(self, client, mock_db):
        resp = client.get("/neo4j/schema")

        assert resp.status_code == 200
        assert "Actor" in resp.json()["schema"]
        mock_db.extract_schema.assert_called()
