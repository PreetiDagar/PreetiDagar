from datetime import datetime, timezone
import math
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="GRIDMIND AI API", version="0.1.0", description="Energy intelligence prototype API. Demo data is synthetic.")
origins = [os.getenv("FRONTEND_ORIGIN", "http://localhost:3000"), "http://localhost:3000"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def build_profile():
    points=[]
    for hour in range(24):
        solar=max(0.0, math.sin((hour-6)/12*math.pi)*68)
        demand=42+18*math.sin((hour-8)/24*math.pi*2)+(18 if 17 < hour < 22 else 0)
        points.append({"time":f"{hour:02d}:00","demand":round(demand,1),"solar":round(solar,1),"baseline":round(demand*1.08,1)})
    return points

@app.get("/")
def root():
    return {"service":"GRIDMIND AI API","status":"ok","data_mode":"synthetic_demo","docs":"/docs"}

@app.get("/health")
def health():
    return {"status":"healthy","timestamp":datetime.now(timezone.utc).isoformat(),"data_mode":"synthetic_demo"}

@app.get("/api/v1/overview")
def overview():
    points=build_profile()
    anomalies=[{"time":"14:00","value":91,"reason":"Consumption spike above expected afternoon profile","severity":"Medium"},{"time":"20:00","value":108,"reason":"Evening peak demand window","severity":"Low"}]
    return {"source":"SIMULATED DEMO DATA","generated_at":datetime.now(timezone.utc).isoformat(),"points":points,"anomalies":anomalies,"recommendation":{"action":"Shift flexible loads toward the solar generation window","reason":"Illustrative schedule aligns flexible demand with midday solar output.","estimated_savings_kwh":18.4}}

class Scenario(BaseModel):
    battery_capacity_kwh: float = Field(80, ge=1, le=10000)
    state_of_charge_pct: float = Field(62, ge=0, le=100)
    flexible_load_pct: float = Field(55, ge=0, le=100)

@app.post("/api/v1/simulate/battery")
def simulate_battery(scenario: Scenario):
    available = scenario.battery_capacity_kwh * scenario.state_of_charge_pct / 100
    shift = (scenario.flexible_load_pct - 50) * 0.22
    return {"available_energy_kwh":round(max(0,min(scenario.battery_capacity_kwh,available+shift)),2),"model":"simplified_illustrative","caveat":"Excludes degradation, inverter limits, tariffs, grid constraints and safety controls."}
