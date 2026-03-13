"""Integration tests — requires a running Neo4j instance and GPU model.

Run with: pytest text2cypher_translator/tests/test_service_integration.py -s -v
"""

from pathlib import Path
import time

import pytest
from fastapi.testclient import TestClient

from text2cypher_translator.service import app

SAMPLE_INIT = str(Path(__file__).resolve().parent.parent.parent / "sample_init.cypher")


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        # Seed the database via the cypher endpoint
        with open(SAMPLE_INIT) as f:
            text = f.read()

        # Clear any existing data
        c.post("/neo4j/cypher", json={"cypher": "MATCH (n) DETACH DELETE n"})

        # Run each statement from the init file
        for stmt in text.split(";"):
            cleaned = "\n".join(
                line for line in stmt.splitlines()
                if line.strip() and not line.strip().startswith("//")
            ).strip()
            if cleaned:
                c.post("/neo4j/cypher", json={"cypher": cleaned})

        yield c

        # Cleanup
        c.post("/neo4j/cypher", json={"cypher": "MATCH (n) DETACH DELETE n"})


class TestSchemaIntegration:
    def test_schema_contains_labels(self, client):
        resp = client.get("/api/schema")

        assert resp.status_code == 200
        schema = resp.json()["schema"]
        print(f"\nSchema:\n{schema}")
        assert "Actor" in schema
        assert "Movie" in schema
        assert "ACTED_IN" in schema


class TestCypherIntegration:
    def test_insert_and_query(self, client):
        client.post(
            "/neo4j/cypher",
            json={
                "cypher": "MERGE (a:Actor {name: $name}) SET a.born = $born",
                "params": {"name": "Keanu Reeves", "born": 1964},
            },
        )

        resp = client.post(
            "/neo4j/cypher",
            json={
                "cypher": "MATCH (a:Actor {name: $name}) RETURN a.name AS name, a.born AS born",
                "params": {"name": "Keanu Reeves"},
            },
        )

        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 1
        assert results[0]["name"] == "Keanu Reeves"

    def test_delete(self, client):
        client.post(
            "/neo4j/cypher",
            json={
                "cypher": "MATCH (a:Actor {name: $name}) DETACH DELETE a",
                "params": {"name": "Keanu Reeves"},
            },
        )

        resp = client.post(
            "/neo4j/cypher",
            json={
                "cypher": "MATCH (a:Actor {name: $name}) RETURN a",
                "params": {"name": "Keanu Reeves"},
            },
        )

        assert resp.json()["results"] == []


class TestNlQueryIntegration:
    def test_natural_language_query(self, client):
        # time to generate and run the cypher query
        start = time.perf_counter()

        resp = client.post(
            "/api/nlquery",
            json={"question": "What movies did Tom Hanks act in?"},
        )

        elapsed = time.perf_counter() - start
        print(f"\nElapsed time: {elapsed:.2f} seconds")

        assert resp.status_code == 200
        body = resp.json()
        print(f"\nQuestion: {body['question']}")
        print(f"Cypher:   {body['cypher']}")
        print(f"Results:  {body['results']}")

        assert "MATCH" in body["cypher"].upper()
        assert len(body["results"]) > 0

    def test_utilization_endpoint(self, client):
        resp = client.get("/api/utilization")
        assert resp.status_code == 200
        util = resp.json()
        print(f"\nResource Utilization: {util}")