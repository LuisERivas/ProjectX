-- cancel_job.lua
--
-- Atomic job cancellation for ProjectX contract.gateway.
--
-- KEYS:
--   [1] job hash key        (e.g. "job:{id}")
--   [2] events stream key   (e.g. "job:{id}:events")
--
-- ARGV:
--   [1] updated_ts          (epoch millis as string)
--   [2] step                (event step label, e.g. "gateway.cancel")
--   [3] event_data_json     (JSON string for canceled event data)
--   [4] error_json          (JSON string stored in job hash `error`)

local job_hash_key = KEYS[1]
local events_stream_key = KEYS[2]

local updated_ts = ARGV[1]
local step = ARGV[2]
local event_data_json = ARGV[3]
local error_json = ARGV[4]

if redis.call("EXISTS", job_hash_key) == 0 then
  return { "missing", "0" }
end

local current_status = redis.call("HGET", job_hash_key, "status") or ""
if current_status == "done" or current_status == "error" or current_status == "canceled" then
  return { current_status, "0" }
end

local ttl_s = tonumber(redis.call("HGET", job_hash_key, "ttl_s"))

redis.call(
  "HSET",
  job_hash_key,
  "status", "canceled",
  "updated_ts", updated_ts,
  "result", "",
  "error", error_json
)

redis.call(
  "XADD",
  events_stream_key,
  "*",
  "type", "canceled",
  "ts", updated_ts,
  "step", step,
  "data", event_data_json
)

if ttl_s ~= nil and ttl_s > 0 then
  redis.call("EXPIRE", job_hash_key, ttl_s)
  redis.call("EXPIRE", events_stream_key, ttl_s)
end

return { "canceled", "1" }
