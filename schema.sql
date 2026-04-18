CREATE DATABASE IF NOT EXISTS engram_obs;

DROP TABLE IF EXISTS engram_obs.lookups;

CREATE TABLE engram_obs.lookups
(
    ts             UInt64,
    request_id     LowCardinality(String),
    request_class  LowCardinality(String),
    layer          UInt8,
    token_pos      UInt16,
    ngram          Array(UInt32),
    bucket         UInt32,
    hit            UInt8
)
ENGINE = MergeTree
ORDER BY (request_class, layer, bucket, ts);
