import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from text2cypher_translator.neo4j_client import Neo4jClient
from text2cypher_translator.text2cypher import Text2CypherTranslator

log = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    cypher: str
    results: list[dict]


class CypherRequest(BaseModel):
    cypher: str
    params: dict = {}

class ResourceUtilization(BaseModel):
    cpu_percent: float
    gpu_percent: float
    gpu_memory_used: int
    gpu_memory_total: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.translator = Text2CypherTranslator(quantize=True)
    app.state.db = Neo4jClient()
    yield
    app.state.db.close()


app = FastAPI(title="Text2Cypher", lifespan=lifespan)


def get_db() -> Neo4jClient:
    return app.state.db


def get_translator() -> Text2CypherTranslator:
    return app.state.translator


@app.post("/api/nlquery", response_model=QueryResponse)
def query(
    req: QueryRequest,
    translator: Text2CypherTranslator = Depends(get_translator),
    db: Neo4jClient = Depends(get_db),
):
    schema = db.extract_schema()
    cypher = translator.generate(req.question, schema)

    try:
        results = db.run(cypher)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cypher execution failed: {e}")

    return QueryResponse(question=req.question, cypher=cypher, results=results)


@app.post("/neo4j/cypher")
def run_cypher(req: CypherRequest, db: Neo4jClient = Depends(get_db)):
    try:
        results = db.run(req.cypher, **req.params)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"cypher": req.cypher, "results": results}


@app.get("/api/schema")
def get_schema(db: Neo4jClient = Depends(get_db)):
    return {"schema": db.extract_schema()}

@app.get("/api/utilization", response_model=ResourceUtilization)
def get_utilization(translator: Text2CypherTranslator = Depends(get_translator)):
    import psutil
    import torch

    cpu_percent = psutil.cpu_percent()
    gpu_percent = 0.0
    gpu_memory_used = 0
    gpu_memory_total = 0

    if torch.cuda.is_available():
        gpu_percent = torch.cuda.utilization()
        gpu_memory_used = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        gpu_memory_total = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert to GB

    return ResourceUtilization(
        cpu_percent=cpu_percent,
        gpu_percent=gpu_percent,
        gpu_memory_used=gpu_memory_used,
        gpu_memory_total=gpu_memory_total,
    )