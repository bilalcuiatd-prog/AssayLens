# AssayLens — complete GitHub and Render deployment

The archive is arranged at **repository root**. After extracting, GitHub must show `requirements.txt`, `Dockerfile`, `render_app.py`, `render.yaml`, the `assaylens` folder, the `web` folder and the `scripts` folder **at the top level**. Do not upload the ZIP as a single file. Do not add another enclosing `AssayLens_AI` folder unless you also configure Render's Root Directory to match.

## 1. Upload source to GitHub

1. Open the GitHub repository `AssayLens_AI`.
2. Select **Add file → Upload files**.
3. Extract the downloaded archive and upload its **contents**, including `assaylens/`, `web/`, `scripts/`, `examples/`, `Dockerfile`, `requirements.txt`, `render_app.py`, `render.yaml`, `.python-version`, and the provided logo.
4. Commit the uploaded files to `main`.
5. Confirm `https://github.com/<account>/AssayLens_AI/blob/main/render_app.py` and `/blob/main/scripts/verify_deploy.py` are accessible to you.

Do not upload `.venv`, `workspace`, API keys, unpublished datasets, or passwords. If GitHub uploads omit hidden files, the main deployment files are still sufficient; `.python-version` and `.dockerignore` are recommended.

## 2A. Preferred Render setup: Python 3

For your existing Render service, select **Settings → Build → Source → Edit** and choose **Python 3** as runtime. This can be changed even if the service previously used Docker. Select branch `main` and keep **Root Directory blank** if the files above are at repository root. Set:

- **Build Command:** `pip install --no-cache-dir -r requirements.txt && python scripts/verify_deploy.py`
- **Start Command:** `uvicorn render_app:app --host 0.0.0.0 --port $PORT --workers 1`
- **Instance:** Free (demo only)

In Environment, set:

- `PYTHON_VERSION` = `3.11.11`
- `ASSAYLENS_SITE_USERNAME` = a username you choose
- `ASSAYLENS_SITE_PASSWORD` = a long unique password

Save settings and deploy the latest commit. `render.yaml` documents this configuration but does **not** automatically reconfigure an existing manually created service.

## 2B. If the existing service remains Docker

Keep **Runtime: Docker**. This archive includes a corrected `Dockerfile` that uses one `COPY . .` rather than several fragile directory copies, includes `render_app.py`, includes the original logo, and starts on Render's `$PORT`. You still must configure both `ASSAYLENS_SITE_USERNAME` and `ASSAYLENS_SITE_PASSWORD` in Render → Environment.

After pushing the files to GitHub, trigger a **Manual Deploy → Deploy latest commit**. If the old Docker build instructions appear, use **Clear build cache & deploy**.

## 3. Verify

`/healthz` returns a small public health status. Opening the site root should ask for credentials; if it returns 503, set username and password in Render Environment. Confirm the logo loads. On Windows you can verify the local repository layout:

`py -3 scripts\verify_deploy.py`

## Limits

This is a password-protected **single-user research demo**, not a production multi-user service. Render Free's local filesystem is temporary. Long jobs, docking executables, untrusted custom-code execution, and PyTDC's full dependency tree are **not supported** by this basic deployment. The primary `requirements.txt` does not install PyTDC. The free service has tight memory and CPU limits; a build or startup failure is possible despite valid configuration. For full scientific workloads, use a more capable private research environment with durable storage and job isolation. Do not store sensitive data in the free ephemeral workspace.
