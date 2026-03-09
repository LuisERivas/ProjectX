# ignoreFUTURESTUFF

## Tasks

- [ ] Add deterministic timeout handling in worker execution wrapper (terminalize or DLQ + ACK).
- [ ] Add poison-message handling for schema/envelope failures (DLQ + ACK).
- [ ] Move DLQ stream key default into shared config and reference it from redis key helper.
