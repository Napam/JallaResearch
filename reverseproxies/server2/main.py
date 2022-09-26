from fastapi import FastAPI

app = FastAPI()

@app.get("/{uri:path}")
async def root(uri) -> str:
    return f"Hello from server 2! Got uri: {uri}"
