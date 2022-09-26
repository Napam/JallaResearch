from fastapi import FastAPI

app = FastAPI()

@app.get("/{uri:path}")
async def root(uri) -> str:
    return f"Hello from server 3! Got uri: {uri}"
