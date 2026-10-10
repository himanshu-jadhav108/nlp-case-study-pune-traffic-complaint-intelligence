# Streamlit Community Cloud Deployment Guide

## 1. Application Overview

**Pune Traffic Complaint Intelligence** is a civic NLP decision-support prototype. It classifies urban traffic grievances into six standard municipal categories (`Congestion`, `Accident`, `Signal`, `Parking`, `Construction`, `Road Condition`) and performs multi-method spatial entity extraction using **spaCy NER (`en_core_web_sm`)** combined with an arterial **Pune Gazetteer Fallback**.

The application is completely self-contained, reading bundled synthetic demonstration data and running local ML/NLP models without external API keys or cloud database requirements.

---

## 2. Streamlit Cloud Configuration & Entry Point

When deploying to [Streamlit Community Cloud](https://share.streamlit.io):

| Setting | Value |
|---|---|
| **Repository** | `<your-github-account>/pune_traffic_complaint_intelligence` |
| **Branch** | `main` (or your active default branch) |
| **Main file path** | `app.py` |
| **Python Version** | `3.11` (Recommended) |

> **Important:** The main file path must be specified exactly as `app.py` in the root of the repository.

---

## 3. Runtime Dependencies

The application runtime dependencies are consolidated in `requirements.txt`:

```text
streamlit>=1.40.0
pandas>=2.2.0
scikit-learn>=1.4.0
plotly>=5.20.0
spacy>=3.8.0,<3.9.0
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

* The spaCy English language model wheel (`en_core_web_sm-3.8.0`) is installed automatically by Streamlit Community Cloud during the container build process via `requirements.txt`.
* Development and automated testing dependencies (e.g. `pytest`) are decoupled in `requirements-dev.txt` to keep cloud deployments fast and lightweight.

---

## 4. Environment Variables & Secrets

* **Secrets Requirement**: **None**. The application operates on bundled local datasets (`data/sample_traffic_complaints.csv` and `data/ner_annotated_test.jsonl`).
* **Secrets Template**: An example file is maintained at `.streamlit/secrets.toml.example` for reference if external civic mapping services or LLMs are added in future iterations.
* **Sensitive File Exclusion**: `.streamlit/secrets.toml` is ignored in `.gitignore`.

---

## 5. Local Startup & Verification Commands

### A. Windows PowerShell (using project's virtual environment)
```powershell
# Verify Python interpreter
& ".\venv\Scripts\python.exe" --version

# Run automated tests
& ".\venv\Scripts\python.exe" -m pytest -q

# Launch Streamlit application locally
& ".\venv\Scripts\python.exe" -m streamlit run app.py
```

### B. macOS / Linux (using project's virtual environment)
```bash
# Verify Python interpreter
./venv/bin/python --version

# Run automated tests
./venv/bin/python -m pytest -q

# Launch Streamlit application locally
./venv/bin/python -m streamlit run app.py
```

---

## 6. Local Validation Audit Results

* **Interpreter**: `.\venv\Scripts\python.exe` (Python 3.14.6)
* **Streamlit**: `1.65.0`
* **Dependency Consistency**: `pip check` verified with **0 broken requirements**.
* **Automated Test Suite**: **11/11 tests passing** (`pytest -q` in 3.5s).
* **Headless Startup**: Streamlit launched headlessly on port `8501`/`8502`; verified HTTP `200 OK` at `/_stcore/health`.
* **Cross-Version Pandas Compatibility**: Verified with `Series.fillna(to_series().astype(str))` ensuring seamless execution on Pandas 2.x and 3.x.

---

## 7. Streamlit Community Cloud Step-by-Step Deployment

1. **Push Repository to GitHub**:
   Ensure all tracked files (`app.py`, `requirements.txt`, `src/`, `data/`, `.streamlit/config.toml`, `README.md`) are pushed to a public or private GitHub repository.
2. **Access Streamlit Cloud**:
   Sign in to [share.streamlit.io](https://share.streamlit.io) using your GitHub account.
3. **Create Application**:
   Click **"Create app"** > **"I already have an app"**.
4. **Configure Repository Parameters**:
   * Select your repository and target branch.
   * Enter `app.py` in the **Main file path** field.
   * Expand **Advanced settings** and ensure Python `3.11` is selected.
5. **Deploy**:
   Click **"Deploy!"**.
   Streamlit Cloud will provision a container, install wheels from `requirements.txt` (including `en_core_web_sm`), and launch the application.

---

## 8. Runtime & Ephemeral Cloud Characteristics

* **Filesystem Ephemerality**: Streamlit Community Cloud runs on ephemeral Linux containers. All datasets and model assets are bundled within the repository tree; no local runtime persistence or SQLite database is required.
* **Memory & Compute Footprint**: The application utilizes lightweight TF-IDF and linear models, consuming under 350 MB of RAM (well below the Community Cloud 1 GB resource limit).
* **Network & API Independence**: The app does not make outbound network requests at runtime, eliminating timeout and firewall hazards on Community Cloud.

---

## 9. Troubleshooting Common Issues

### Issue 1: "spaCy model en_core_web_sm not found"
* **Cause**: Missing model wheel in requirements.
* **Resolution**: Verify that line `https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl` is present in `requirements.txt`. The application also contains an automatic fallback to the built-in Pune arterial gazetteer so the app never crashes if the spaCy model fails to initialize.

### Issue 2: "Pandas TypeError: 'value' parameter must be a scalar, dict or Series"
* **Cause**: Incompatible index casting during `fillna` in newer pandas versions.
* **Resolution**: Already resolved in `src/data_utils.py` using `frame.index.to_series().astype(str)`.

### Issue 3: "Port or Headless warning"
* **Cause**: Local configuration interfering with cloud host settings.
* **Resolution**: The provided `.streamlit/config.toml` includes standard `headless = true` and `enableXsrfProtection = true` tokens designed for cloud environments.
