import pytest

from text2cypher_translator.text2cypher import Text2CypherTranslator
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture(scope="session")
def translator():
    return Text2CypherTranslator(quantize=True)


def test_generates_valid_cypher(translator):
    schema = "(:Actor {name: string})-[:ACTED_IN]->(:Movie {title: string})"
    question = "What movies did Tom Hanks act in?"

    cypher = translator.generate(question, schema)

    print(f"Generated cypher: {cypher}")

    assert "MATCH" in cypher.upper()
    assert len(cypher) > 0
