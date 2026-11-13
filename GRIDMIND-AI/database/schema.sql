-- GRIDMIND AI PostgreSQL starter schema
-- Run in the Supabase SQL editor after creating a project.
create extension if not exists pgcrypto;

create table if not exists energy_sites (
  id uuid primary key default gen_random_uuid(),
  site_name text not null,
  site_type text not null default 'building',
  timezone_name text not null default 'Asia/Kolkata',
  created_at timestamptz not null default now()
);

create table if not exists energy_readings (
  id bigserial primary key,
  site_id uuid not null references energy_sites(id) on delete cascade,
  recorded_at timestamptz not null,
  demand_kwh numeric(12,3),
  solar_generation_kwh numeric(12,3),
  battery_soc_pct numeric(5,2),
  source text not null default 'synthetic_demo',
  created_at timestamptz not null default now(),
  unique(site_id, recorded_at)
);
create index if not exists idx_energy_readings_site_time on energy_readings(site_id, recorded_at desc);

create table if not exists energy_scenarios (
  id uuid primary key default gen_random_uuid(),
  site_id uuid references energy_sites(id) on delete set null,
  scenario_name text not null,
  parameters jsonb not null default '{}'::jsonb,
  results jsonb not null default '{}'::jsonb,
  data_mode text not null default 'synthetic_demo',
  created_at timestamptz not null default now()
);
