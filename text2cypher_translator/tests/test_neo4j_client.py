import os
from pathlib import Path

import pytest

from text2cypher_translator.neo4j_client import Neo4jClient

SAMPLE_INIT = str(Path(__file__).resolve().parent.parent.parent / "sample_init.cypher")


@pytest.fixture(scope="session")
def db():
    client = Neo4jClient()
    # Start clean
    client.run("MATCH (n) DETACH DELETE n")
    yield client
    client.close()


class TestNeo4jClient:
    def test_init_db(self, db):
        schema = db.init_db(SAMPLE_INIT)

        print(f"\nExtracted schema:\n{schema}")

        assert "Node properties:" in schema
        assert "**Actor**" in schema
        assert "**Movie**" in schema
        assert "**Director**" in schema
        assert "The relationships:" in schema
        assert "ACTED_IN" in schema
        assert "DIRECTED" in schema

    def test_extract_schema(self, db):
        schema = db.extract_schema()

        assert "`name`: STRING" in schema
        assert "`title`: STRING" in schema

    def test_insert_and_query(self, db):
        db.run(
            "MERGE (a:Actor {name: $name}) SET a.born = $born",
            name="Keanu Reeves",
            born=1964,
        )

        results = db.run(
            "MATCH (a:Actor {name: $name}) RETURN a.name AS name, a.born AS born",
            name="Keanu Reeves",
        )

        assert len(results) == 1
        assert results[0]["name"] == "Keanu Reeves"
        assert results[0]["born"] == 1964

    def test_query_relationships(self, db):
        results = db.run(
            "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) "
            "RETURN a.name AS actor, m.title AS movie "
            "ORDER BY actor, movie"
        )

        actors = {r["actor"] for r in results}
        assert "Tom Hanks" in actors
        assert "Meg Ryan" in actors

    def test_clear_db(self, db):
        db.run("MATCH (n) DETACH DELETE n")

        results = db.run("MATCH (n) RETURN count(n) AS count")
        assert results[0]["count"] == 0
