-- Rule 1: transaction velocity (5+ transactions in a rolling 10-minute window)
WITH velocity AS (
    SELECT *,
        COUNT(*) OVER (
            PARTITION BY customer_id
            ORDER BY transaction_ts
            RANGE BETWEEN INTERVAL 10 MINUTES PRECEDING AND CURRENT ROW
        ) AS txns_last_10m
    FROM transactions
)
SELECT transaction_id, customer_id, transaction_ts, amount, txns_last_10m
FROM velocity
WHERE txns_last_10m >= 5
ORDER BY transaction_ts DESC;

-- Rule 2: possible structuring (multiple near-threshold cash deposits in one day)
WITH daily_cash AS (
    SELECT customer_id, CAST(transaction_ts AS DATE) AS txn_date,
           COUNT(*) AS deposit_count, SUM(amount) AS daily_total
    FROM transactions
    WHERE transaction_type = 'CASH_DEPOSIT'
      AND amount BETWEEN 8000 AND 9999.99
    GROUP BY customer_id, CAST(transaction_ts AS DATE)
)
SELECT *
FROM daily_cash
WHERE deposit_count >= 3 AND daily_total >= 20000
ORDER BY daily_total DESC;

-- Rule 3: geographic inconsistency between merchant and IP country
SELECT transaction_id, customer_id, transaction_ts, amount,
       merchant_country, ip_country
FROM transactions
WHERE merchant_country <> ip_country
  AND merchant_country <> 'UNKNOWN'
  AND ip_country <> 'UNKNOWN';

-- Rule 4: high-value transactions (review threshold is illustrative)
SELECT transaction_id, customer_id, transaction_ts, amount, transaction_type
FROM transactions
WHERE amount >= 10000
ORDER BY amount DESC;
