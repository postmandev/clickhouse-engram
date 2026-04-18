-- Every query here runs against the same MergeTree table, ordered by
-- (request_class, layer, bucket, ts). Numbers and timings in README.md
-- correspond to a run with 10.16M events.

-- Hit rate by request class — workload utilization varies 8×
SELECT
    request_class,
    round(avg(hit), 3) AS hit_rate,
    count() AS lookups
FROM engram_obs.lookups
GROUP BY request_class
ORDER BY hit_rate DESC;

-- Pareto / Lorenz curve of hit concentration
-- (one row per hit bucket, sorted hottest first, cumulative lookup share)
WITH per_bucket AS (
    SELECT bucket, count() AS hits
    FROM engram_obs.lookups WHERE hit = 1 GROUP BY bucket
)
SELECT
    row_number() OVER (ORDER BY hits DESC) / count() OVER () AS bucket_frac,
    sum(hits) OVER (ORDER BY hits DESC) / sum(hits) OVER () AS cum_hit_frac
FROM per_bucket
ORDER BY hits DESC;

-- Concentration knees: what share of hits does the top N% capture?
WITH per_bucket AS (
    SELECT bucket, count() AS hits
    FROM engram_obs.lookups WHERE hit = 1 GROUP BY bucket
),
ranked AS (
    SELECT hits,
           row_number() OVER (ORDER BY hits DESC) AS rnk,
           count() OVER () AS n,
           sum(hits) OVER () AS total
    FROM per_bucket
)
SELECT
    round(sumIf(hits, rnk <= n * 0.001) / any(total), 3) AS top_0_1pct,
    round(sumIf(hits, rnk <= n * 0.01)  / any(total), 3) AS top_1pct,
    round(sumIf(hits, rnk <= n * 0.10)  / any(total), 3) AS top_10pct
FROM ranked;

-- Temperature distribution of populated buckets
-- (replace 73771 with the populated count reported by driver.py)
WITH per_bucket AS (
    SELECT bucket, count() AS hits
    FROM engram_obs.lookups WHERE hit = 1 GROUP BY bucket
)
SELECT
    countIf(hits >= 1000)                AS blazing,
    countIf(hits >= 100  AND hits < 1000) AS hot,
    countIf(hits >= 10   AND hits < 100)  AS warm,
    countIf(hits >= 1    AND hits < 10)   AS cold,
    73771 - count()                      AS dead
FROM per_bucket;
