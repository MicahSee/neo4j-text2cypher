import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

_TYPE_MAP = {
    "String": "STRING",
    "Long": "INTEGER",
    "Integer": "INTEGER",
    "Float": "FLOAT",
    "Double": "FLOAT",
    "Boolean": "BOOLEAN",
    "Date": "DATE",
    "DateTime": "DATE_TIME",
    "ZonedDateTime": "DATE_TIME",
    "LocalDateTime": "DATE_TIME",
    "Point": "POINT",
    "StringArray": "LIST(STRING)",
}


class Neo4jClient:
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        self._uri = uri or os.environ["NEO4J_URI"]
        self._user = user or os.environ["NEO4J_USER"]
        self._password = password or os.environ["NEO4J_PASSWORD"]
        self._driver = GraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )
        self._driver.verify_connectivity()
        log.info("Connected to %s", self._uri)

    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def init_db(self, cypher_file: str) -> str:
        """Load a .cypher file into Neo4j and return the extracted schema."""
        statements = self._parse_cypher_file(cypher_file)
        log.info("Parsed %d statement(s) from %s", len(statements), cypher_file)

        with self._driver.session() as session:
            for i, stmt in enumerate(statements, 1):
                log.info("[%d/%d] %s", i, len(statements), stmt.replace("\n", " ")[:120])
                session.run(stmt)

        log.info("Database initialisation complete.")
        return self.extract_schema()

    def extract_schema(self) -> str:
        """Query the live database and return a formatted schema string."""
        lines: list[str] = []

        with self._driver.session() as session:
            # Node properties
            node_props = session.run(
                "CALL db.schema.nodeTypeProperties() "
                "YIELD nodeLabels, propertyName, propertyTypes "
                "RETURN nodeLabels, propertyName, propertyTypes"
            ).data()

            label_props: dict[str, list[dict]] = {}
            for row in node_props:
                label = row["nodeLabels"][0]
                label_props.setdefault(label, []).append(row)

            lines.append("Node properties:")
            for label in sorted(label_props):
                lines.append(f"- **{label}**")
                for prop in label_props[label]:
                    name = prop["propertyName"]
                    raw_type = prop["propertyTypes"][0] if prop["propertyTypes"] else "STRING"
                    display_type = _TYPE_MAP.get(raw_type, raw_type.upper())
                    detail = self._property_detail(session, label, name, display_type)
                    lines.append(f"  - `{name}`: {display_type}{detail}")

            # Relationship properties
            rel_props = session.run(
                "CALL db.schema.relTypeProperties() "
                "YIELD relType, propertyName, propertyTypes "
                "RETURN relType, propertyName, propertyTypes"
            ).data()

            rel_type_props: dict[str, list[dict]] = {}
            for row in rel_props:
                rel = row["relType"].strip(":`")
                if row["propertyName"]:
                    rel_type_props.setdefault(rel, []).append(row)

            lines.append("Relationship properties:")
            if rel_type_props:
                for rel in sorted(rel_type_props):
                    lines.append(f"- **{rel}**")
                    for prop in rel_type_props[rel]:
                        name = prop["propertyName"]
                        raw_type = prop["propertyTypes"][0] if prop["propertyTypes"] else "STRING"
                        display_type = _TYPE_MAP.get(raw_type, raw_type.upper())
                        lines.append(f"  - `{name}`: {display_type}")

            # Relationship patterns
            patterns = session.run(
                "MATCH (a)-[r]->(b) "
                "WITH head(labels(a)) AS from_l, type(r) AS rel, head(labels(b)) AS to_l "
                "RETURN DISTINCT from_l, rel, to_l "
                "ORDER BY from_l, rel, to_l"
            ).data()

            lines.append("The relationships:")
            for p in patterns:
                lines.append(f"(:{p['from_l']})-[:{p['rel']}]->(:{p['to_l']})")

        return "\n".join(lines)

    # ── Query execution ──────────────────────────────────────────

    def run(self, cypher: str, **params) -> list[dict]:
        """Execute a single Cypher statement and return the results as dicts."""
        with self._driver.session() as session:
            return session.run(cypher, params).data()

    # ── Private helpers ──────────────────────────────────────────

    @staticmethod
    def _parse_cypher_file(path: str) -> list[str]:
        # check valid path
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Cypher file not found: {path}")
        text = Path(path).read_text(encoding="utf-8")
        statements = []
        for stmt in text.split(";"):
            cleaned = "\n".join(
                line for line in stmt.splitlines()
                if line.strip() and not line.strip().startswith("//")
            ).strip()
            if cleaned:
                statements.append(cleaned)
        return statements

    @staticmethod
    def _property_detail(session, label: str, prop: str, display_type: str) -> str:
        if display_type == "POINT":
            return ""

        if display_type in ("INTEGER", "FLOAT", "DATE_TIME", "DATE"):
            result = session.run(
                f"MATCH (n:`{label}`) WHERE n.`{prop}` IS NOT NULL "
                f"RETURN min(n.`{prop}`) AS mn, max(n.`{prop}`) AS mx"
            ).single()
            if result and result["mn"] is not None:
                return f" Min: {result['mn']}, Max: {result['mx']}"
            return ""

        result = session.run(
            f"MATCH (n:`{label}`) WHERE n.`{prop}` IS NOT NULL "
            f"RETURN n.`{prop}` AS val LIMIT 1"
        ).single()
        if result and result["val"] is not None:
            return f' Example: "{result["val"]}"'
        return ""


