-- mark_running.lua
--
-- Atomic running-transition script for ProjectX contract.worker.
--
-- Guarantees:
--   1) Job status transition to "running" is conditional and atomic.
--   2) "running" event emission is atomic with status update.
--   3) TTL is consistently applied to both job hash and events stream.
--   4) Non-runnable deliveries are ACKed immediately.
--
-- KEYS:
--   [1] job hash key       (e.g. "job:{id}")
--   [2] events stream key  (e.g. "job:{id}:events")
--   [3] queue stream key   (e.g. "jobs:stream")
--
-- ARGV:
--   [1] worker group       (e.g. "workers")
--   [2] msg id             (Redis stream id to XACK when skipping)
--   [3] default_ttl_s      (fallback ttl when hash ttl_s is missing/invalid)
--   [4] updated_ts         (epoch millis as string)
--   [5] step               (event step label, e.g. "worker.start")
--   [6] event_data_json    (JSON string for event `data` field)

local job_key = KEYS[1]
local events_key = KEYS[2]
local queue_stream_key = KEYS[3]

local worker_group = ARGV[1]
local msg_id = ARGV[2]
local default_ttl_s = tonumber(ARGV[3])
local updated_ts = ARGV[4]
local step = ARGV[5]
local event_data_json = ARGV[6]

if redis.call("EXISTS", job_key) == 0 then
  redis.call("XACK", queue_stream_key, worker_group, msg_id)
  return { "skip", "0", "missing" }
end

local status = redis.call("HGET", job_key, "status") or ""
if status ~= "queued" then
  redis.call("XACK", queue_stream_key, worker_group, msg_id)
  return { "skip", "0", status }
end

local ttl_raw = redis.call("HGET", job_key, "ttl_s")
local ttl_s = tonumber(ttl_raw)
if ttl_s == nil or ttl_s <= 0 then
  ttl_s = default_ttl_s or 0
end

redis.call(
  "HSET",
  job_key,
  "status", "running",
  "updated_ts", updated_ts,
  "error", "",
  "result", ""
)

redis.call(
  "XADD",
  events_key,
  "*",
  "type", "running",
  "ts", updated_ts,
  "step", step,
  "data", event_data_json
)

if ttl_s ~= nil and ttl_s > 0 then
  redis.call("EXPIRE", job_key, ttl_s)
  redis.call("EXPIRE", events_key, ttl_s)
end

return { "run", tostring(ttl_s), "queued" }
