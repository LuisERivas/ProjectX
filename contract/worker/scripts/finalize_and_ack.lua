-- finalize_and_ack.lua
--
-- Atomic finalization script for ProjectX contract.worker.
--
-- Responsibilities (see ProjectX/BOUNDARY_MATRIX.md):
--   1. Update job hash `job:{id}` to terminal state (`status`, `updated_ts`,
--      `result`, `error`).
--   2. Emit a terminal event into `job:{id}:events`.
--   3. Apply TTL to both the job hash and events stream.
--   4. ACK the original queue message from `jobs:stream`.
--
-- This guarantees the invariant: terminal state + terminal event + TTL
-- are all applied before XACK, and the whole sequence is atomic.
--
-- KEYS:
--   [1] job hash key          (e.g. "job:{id}")
--   [2] events stream key     (e.g. "job:{id}:events")
--   [3] queue stream key      (e.g. "jobs:stream")
--
-- ARGV:
--   [1] group name            (e.g. "workers")
--   [2] msg id                (Redis stream id to XACK)
--   [3] ttl_s                 (integer seconds; "0" or empty disables TTL)
--   [4] terminal status       ("done" | "error" | "canceled")
--   [5] updated_ts            (millis since epoch as string)
--   [6] step                  (event step label, e.g. "worker")
--   [7] result_json           (JSON string or "" to clear)
--   [8] error_json            (JSON string or "" to clear)
--   [9] event_data_json       (JSON string for event `data` field)

local job_key = KEYS[1]
local events_key = KEYS[2]
local stream_key = KEYS[3]

local group = ARGV[1]
local msg_id = ARGV[2]
local ttl_s_raw = ARGV[3]
local status = ARGV[4]
local updated_ts = ARGV[5]
local step = ARGV[6]
local result_json = ARGV[7]
local error_json = ARGV[8]
local event_data_json = ARGV[9]

local ttl_s = tonumber(ttl_s_raw)

-- Missing job guard: if hash no longer exists (e.g., expired/deleted), do not
-- recreate a partial terminal hash/event shape. ACK and exit.
if redis.call("EXISTS", job_key) == 0 then
  redis.call("XACK", stream_key, group, msg_id)
  return 0
end

-- Cancellation precedence: if an out-of-band cancel already marked this job
-- terminal, do not overwrite it with done/error. Just ACK the queue message.
local current_status = redis.call("HGET", job_key, "status")
if current_status == "canceled" and status ~= "canceled" then
  if ttl_s ~= nil and ttl_s > 0 then
    redis.call("EXPIRE", job_key, ttl_s)
    redis.call("EXPIRE", events_key, ttl_s)
  end
  redis.call("XACK", stream_key, group, msg_id)
  return 2
end

-- Build HSET arguments for the job hash.
local hset_args = { "status", status, "updated_ts", updated_ts }

if result_json ~= nil and result_json ~= "" then
  table.insert(hset_args, "result")
  table.insert(hset_args, result_json)
end

if error_json ~= nil and error_json ~= "" then
  table.insert(hset_args, "error")
  table.insert(hset_args, error_json)
end

-- If caller explicitly passes empty strings for result/error, we still want
-- to overwrite previous values to maintain a consistent schema.
if result_json == "" then
  table.insert(hset_args, "result")
  table.insert(hset_args, "")
end

if error_json == "" then
  table.insert(hset_args, "error")
  table.insert(hset_args, "")
end

redis.call("HSET", job_key, unpack(hset_args))

if ttl_s ~= nil and ttl_s > 0 then
  redis.call("EXPIRE", job_key, ttl_s)
end

-- Terminal event entry.
local ev_fields = {
  "type", status,
  "ts", updated_ts,
  "step", step,
  "data", event_data_json,
}

redis.call("XADD", events_key, "*", unpack(ev_fields))

if ttl_s ~= nil and ttl_s > 0 then
  redis.call("EXPIRE", events_key, ttl_s)
end

-- Finally ACK the message from the queue stream.
redis.call("XACK", stream_key, group, msg_id)

return 1
