from fastapi import FastAPI, HTTPException
import subprocess

app = FastAPI()

@app.get("/run_transferencia")
def run_transferencia(username: str):
    try:
        command = ["python", "Transferencia.py", username]
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        output, error = process.communicate()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error running Transferencia.py script: " + str(e))

    if error:
        raise HTTPException(status_code=500, detail="Error running Transferencia.py script: " + error.decode("utf-8"))

    return f"Transferencia.py script has been run. Output: {output.decode('utf-8')}"

@app.get("/run_sutura")
def run_sutura(username: str):
    try:
        command = ["python", "Sutura.py", username]
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        output, error = process.communicate()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error running Sutura.py script: " + str(e))

    if error:
        raise HTTPException(status_code=500, detail="Error running Sutura.py script: " + error.decode("utf-8"))

    return f"Sutura.py script has been run. Output: {output.decode('utf-8')}"

@app.get("/run_corte")
def run_corte(username: str):
    try:
        command = ["python", "Corte.py", username]
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        output, error = process.communicate()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error running Corte.py script: " + str(e))

    if error:
        raise HTTPException(status_code=500, detail="Error running Corte.py script: " + error.decode("utf-8"))

    return f"Corte.py script has been run. Output: {output.decode('utf-8')}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8900)
     