# Run Lumina CV

Extract the ZIP and open the `Lumina-CV` folder. The trained model is included; no dataset download or training is needed.

## Requirements

- Python 3.12
- Node.js 22 with npm
- Internet access for the first dependency installation

The commands below use Windows PowerShell.

## 1. Set up the backend (first time only)

Open a terminal in `Lumina-CV`:

```powershell
cd core
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[api]"
Copy-Item .env.example .env
```

## 2. Start the backend

In the same terminal, inside `core`:

```powershell
$env:LUMINA_PORT = "8001"
.\.venv\Scripts\python.exe -m nta_zerodce.api
```

Keep this terminal running. Open http://127.0.0.1:8001/api/v1/health and check that `model_loaded` is `true`.

## 3. Start the web application

Open a second terminal in `Lumina-CV`:

```powershell
cd web
npm.cmd ci
$env:LUMINA_API_URL = "http://127.0.0.1:8001"
npm.cmd run dev -- --port 3000
```

Run `npm.cmd ci` only on the first setup or after dependencies change.

## 4. Use the application

Open http://localhost:3000. Choose a PNG or JPEG, click **Enhance**, compare the images, and use **Download PNG** to save the result.

Keep both terminals running. Press `Ctrl+C` in each terminal to stop the application.

If port 8001 is occupied, choose another free port in the backend's `LUMINA_PORT` and use the same port in the web terminal's `LUMINA_API_URL`. Restart both servers. If port 3000 is occupied, start the web app with `--port 3001` and open http://localhost:3001 instead.
