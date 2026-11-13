# GRIDMIND AI — Energy Intelligence & Digital Twin

A full-stack AI energy intelligence platform for forecasting building demand and solar generation, detecting unusual consumption, and simulating battery dispatch decisions.

> **Prototype data notice:** The initial version uses a deterministic, clearly labelled synthetic energy dataset. It does not connect to live grid telemetry or control physical equipment. Forecasts and recommendations are decision-support demonstrations, not operational instructions.

## Architecture
- **Frontend:** Next.js App Router, React, TypeScript, Recharts, Lucide — deploy to Vercel.
- **API:** Python, FastAPI, Pandas, NumPy, scikit-learn — deploy to Render.
- **Database:** PostgreSQL — Supabase.
- **Source control:** GitHub.

## Current build status
- [x] Initial monorepo scaffold and deployment configuration
- [x] Interactive energy command centre with responsive charts
- [x] FastAPI demo endpoints for telemetry, forecast, anomaly and battery simulation
- [x] PostgreSQL schema draft
- [ ] Train and validate forecasting/anomaly models on documented public datasets
- [ ] Persist projects and user scenarios in Supabase
- [ ] Configure Vercel, Render and Supabase project secrets and deployments

## Run locally

### API
```bash
cd GRIDMIND-AI/backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd GRIDMIND-AI/frontend
npm install
npm run dev
```
Open http://localhost:3000. Set `NEXT_PUBLIC_API_URL=http://localhost:8000` in `frontend/.env.local` to use the API.

## Deploy
1. Import this GitHub repository into Vercel; set Root Directory to `GRIDMIND-AI/frontend`.
2. Import the same repository into Render as a Web Service; set Root Directory to `GRIDMIND-AI/backend`, Build Command to `pip install -r requirements.txt`, Start Command to `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
3. Create a Supabase PostgreSQL project and run `database/schema.sql` in its SQL editor.
4. Set `NEXT_PUBLIC_API_URL` in Vercel to the deployed Render API URL. Set `FRONTEND_ORIGIN` in Render to the deployed Vercel URL.
5. Redeploy both services.

## Responsible use
The battery simulator is a simplified illustrative model. It omits inverter limits, battery degradation, tariffs, grid constraints, safety protections and local regulatory requirements. Do not use it to operate real energy equipment.
