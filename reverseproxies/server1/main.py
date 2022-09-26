from fastapi import FastAPI

app = FastAPI()

@app.get("/{uri:path}")
async def root(uri) -> str:
    return f"Hello from server 1! Got uri: {uri}"
