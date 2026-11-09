# How to upload to GitHub and deploy on Streamlit

Streamlit Community Cloud does **not** take a separate upload. It deploys straight from a GitHub repository.
So there is **one repository**: you push this folder to GitHub, then point Streamlit at `app.py` in it.

## 1. What goes to GitHub

Upload everything in this folder **except** what `.gitignore` excludes:

| Path | Upload? | Why |
|---|---|---|
| `app.py` | ✅ | the dashboard |
| `requirements.txt` | ✅ | Streamlit Cloud installs these |
| `.streamlit/config.toml` | ✅ | dashboard theme |
| `data/dashboard/` (all files) | ✅ | the only data the dashboard reads (~15 MB) |
| `notebooks/Road_Accident_Spatiotemporal_Severity.ipynb` | ✅ | full analysis with outputs |
| `outputs/figures/*.png` | ✅ | figures for the report and README |
| `documentation/` | ✅ | metrics, rules, data-quality summary |
| `README.md`, `DEPLOYMENT.md`, `LICENSE_DATA.md`, `.gitignore`, `requirements-notebook.txt` | ✅ | |
| `data/raw/*.csv`, `*.geojson`, `*.xlsx` | ❌ | 115 MB; the notebook downloads them automatically |
| `data/processed/*.geoparquet/` | ❌ | large Spark intermediates, re-created by the notebook |
| `models/` | ❌ | Spark model folders, re-created by the notebook |

The largest uploaded file is `collisions_points.parquet` (well under GitHub's 100 MB per-file limit), so
Git LFS is not needed.

### Option A – GitHub website (no command line)
1. Sign in at github.com → **New repository** → name it e.g. `road-accident-hotspots` → Public → **Create**.
2. Click **uploading an existing file**. Drag in the files and folders listed above. The browser uploader
   accepts folders, but upload `data/dashboard` as a folder so its path is kept.
3. **Commit changes**.

### Option B – Git command line (from this folder)
```bash
git init
git add .
git commit -m "Road accident hotspot and severity project"
git branch -M main
git remote add origin https://github.com/<your-username>/road-accident-hotspots.git
git push -u origin main
```
`git add .` respects `.gitignore`, so the raw data and Spark folders are skipped automatically.

## 2. Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub.
2. **Create app** → **Deploy a public app from GitHub**.
3. Repository: `<your-username>/road-accident-hotspots` · Branch: `main` · Main file path: `app.py`.
4. **Advanced settings → Python version: 3.11**.
5. **Deploy**. The first build takes a few minutes (installing `requirements.txt`). You then get a public URL
   such as `https://road-accident-hotspots.streamlit.app` to show your faculty.

If you change the notebook and re-run it, commit the new `data/dashboard/` files and Streamlit redeploys
automatically.

## 3. Run the dashboard locally (Windows)
```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```
Open http://localhost:8501.

## 4. Re-run the analysis notebook
Needs Java 17 and Python 3.11 (Linux, WSL2 or the university DGX):
```bash
pip install -r requirements-notebook.txt
cd notebooks
jupyter lab Road_Accident_Spatiotemporal_Severity.ipynb
```
Optional environment variables: `SPARK_MASTER` (default `local[4]`), `SPARK_DRIVER_MEMORY` (default `3g`),
`POSTGIS_URL` / `POSTGIS_USER` / `POSTGIS_PASSWORD` for the PostGIS step (skipped if no server is reachable),
`ROAD_RAW_DIR` to read the raw files from another folder.

## Troubleshooting
| Symptom | Fix |
|---|---|
| Streamlit Cloud: `ModuleNotFoundError` | check `requirements.txt` is in the repository root |
| Streamlit Cloud: error loading `severity_predictor.joblib` | keep `scikit-learn==1.9.1` pinned (the model must be loaded with the version that saved it) |
| "data/dashboard/ is missing" | upload the `data/dashboard` folder with its path intact |
| Map tiles blank | the basemap loads from Carto; check the network, as the data layers still render |
