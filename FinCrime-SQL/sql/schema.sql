-- FinCrime SQL | Demonstration relational schema
-- All records in the public demo are synthetic.
CREATE TABLE customers (
    customer_id VARCHAR PRIMARY KEY,
    customer_name VARCHAR,
    customer_segment VARCHAR,
    risk_tier VARCHAR,
    home_country VARCHAR,
    onboarding_date DATE
);

CREATE TABLE accounts (
    account_id VARCHAR PRIMARY KEY,
    customer_id VARCHAR REFERENCES customers(customer_id),
    account_type VARCHAR,
    currency VARCHAR,
    opened_date DATE,
    account_status VARCHAR
);

CREATE TABLE transactions (
    transaction_id VARCHAR PRIMARY KEY,
    account_id VARCHAR REFERENCES accounts(account_id),
    customer_id VARCHAR REFERENCES customers(customer_id),
    transaction_ts TIMESTAMP,
    amount DECIMAL(14,2),
    currency VARCHAR,
    transaction_type VARCHAR,
    channel VARCHAR,
    merchant_category VARCHAR,
    merchant_country VARCHAR,
    ip_country VARCHAR,
    device_id VARCHAR,
    counterparty_id VARCHAR,
    synthetic_scenario VARCHAR
);

CREATE TABLE alerts (
    alert_id VARCHAR PRIMARY KEY,
    transaction_id VARCHAR REFERENCES transactions(transaction_id),
    customer_id VARCHAR,
    alert_rule VARCHAR,
    risk_score INTEGER,
    severity VARCHAR,
    created_at TIMESTAMP,
    alert_status VARCHAR
);

CREATE TABLE investigations (
    case_id VARCHAR PRIMARY KEY,
    alert_id VARCHAR REFERENCES alerts(alert_id),
    assigned_to VARCHAR,
    case_status VARCHAR,
    disposition VARCHAR,
    opened_at TIMESTAMP,
    notes VARCHAR
);
