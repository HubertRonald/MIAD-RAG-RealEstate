from fastapi import FastAPI

app = FastAPI(title="MIAD RAG Backend")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}