from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def inicio():
    return {"mensaje": "Inicio"}

@app.get("/usuarios")
def usuarios():
    return ["Juan", "Ana"]

@app.get("/productos")
def productos():
    return ["Laptop", "Mouse"]