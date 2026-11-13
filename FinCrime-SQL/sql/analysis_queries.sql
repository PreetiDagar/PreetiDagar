-- 1. Customer-level transaction concentration
SELECT customer_id, COUNT(*) AS transaction_count,
       ROUND(SUM(amount),2) AS total_value,
       ROUND(AVG(amount),2) AS average_ticket
FROM transactions
GROUP BY customer_id
ORDER BY total_value DESC
LIMIT 25;

-- 2. Daily transaction velocity
SELECT CAST(transaction_ts AS DATE) AS txn_day,
       COUNT(*) AS transaction_count,
       ROUND(SUM(amount),2) AS total_value,
       COUNT(DISTINCT customer_id) AS active_customers
FROM transactions
GROUP BY 1
ORDER BY 1;

-- 3. Channel mix and average ticket
SELECT channel, COUNT(*) AS transaction_count,
       ROUND(SUM(amount),2) AS total_value,
       ROUND(AVG(amount),2) AS average_ticket
FROM transactions
GROUP BY channel
ORDER BY total_value DESC;

-- 4. Counterparty concentration
SELECT counterparty_id, COUNT(*) AS transaction_count,
       COUNT(DISTINCT customer_id) AS distinct_customers,
       ROUND(SUM(amount),2) AS total_value
FROM transactions
GROUP BY counterparty_id
HAVING COUNT(DISTINCT customer_id) >= 3
ORDER BY total_value DESC;

-- 5. Customer activity bands using a CTE and CASE
WITH customer_activity AS (
    SELECT customer_id, COUNT(*) AS txn_count, SUM(amount) AS total_value
    FROM transactions GROUP BY customer_id
)
SELECT customer_id, txn_count, ROUND(total_value,2) AS total_value,
       CASE WHEN txn_count >= 30 THEN 'High activity'
            WHEN txn_count >= 12 THEN 'Moderate activity'
            ELSE 'Lower activity' END AS activity_band
FROM customer_activity
ORDER BY txn_count DESC;

-- 6. Hour-of-day monitoring profile
SELECT EXTRACT(HOUR FROM transaction_ts) AS transaction_hour,
       COUNT(*) AS transaction_count,
       ROUND(AVG(amount),2) AS average_amount,
       ROUND(SUM(amount),2) AS total_value
FROM transactions
GROUP BY 1
ORDER BY 1;

-- 7. Geographic mismatch review signals
SELECT transaction_id, customer_id, transaction_ts,
       merchant_country, ip_country, amount
FROM transactions
WHERE merchant_country <> ip_country
ORDER BY transaction_ts DESC;

-- 8. Rank customers within segment using a window function
WITH segment_value AS (
    SELECT c.customer_segment, t.customer_id, SUM(t.amount) AS total_value
    FROM transactions t JOIN customers c USING(customer_id)
    GROUP BY c.customer_segment, t.customer_id
)
SELECT customer_segment, customer_id, ROUND(total_value,2) AS total_value,
       DENSE_RANK() OVER(PARTITION BY customer_segment ORDER BY total_value DESC) AS segment_rank
FROM segment_value
ORDER BY customer_segment, segment_rank;
