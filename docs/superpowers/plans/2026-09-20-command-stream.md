# Command stream and Gym execution implementation plan

**Goal:** Implement the user-approved independent 50 Hz writer, watchdog and Gym execution path without enabling hardware motion.

**Architecture:** `command_stream.py` owns one writer and one independent watchdog. Immutable targets carry sequence IDs and monotonic leases. `motion_backend.py` combines explicit task limits, verified observation guards, command acknowledgement and successor observations into StepResult. Existing GdkCommandPort remains the only SDK sender.

**Spec:** User request to complete sender/watchdog/Gym; safety boundaries in `docs/g2-adaptation-status.md`.

**Constraints:** No hardware commands during implementation. Do not change modes, user staging or reference project. Cannot cancel an in-flight SDK call in Python; bounded stop must report failure and must not release SDK resources while writer is alive. Test with isolated ports/readers.

## Tasks (inline execution, checkpoints after tests)

- [x] Write `tests/test_g2_command_stream.py`: repeated sends, acknowledged sequence, lease expiry, blocking send detected by watchdog, bounded stop, no automatic restart. Run failing tests.
- [x] Implement `CommandStream.submit(target) -> sequence`, `wait_sent(sequence, timeout) -> monotonic_time`, `check()`, `stop()` in `g2_local/command_stream.py`. All sends occur on one thread; watchdog only marks faults and signals stop. Worker owns port.stop after its in-flight call ends.
- [x] Run stream tests; verify no send occurs after successful stop and late send never becomes a successful acknowledgement.
- [x] Write `tests/test_g2_motion_backend.py`: default deny, clipping/action acknowledgement, observed successor, stale observation/reader exception stopping, Gym terminal stop and no automatic rearm. Run failing tests.
- [x] Implement `MotionBackend(reader, port, config, observation_guard, outcome, ...)`: validate explicit config; observe/check; plan immutable pose; submit/wait; observe/check after send; return StepResult. Any exception invalidates backend and stops stream. Reader.close only after stream has stopped.
- [x] Run targeted tests, full suite and diff checks; read-only review of concurrency and lifecycle; document commissioning gaps and tests. No commit or hardware enable requested.

Result: 76 tests passed (2 existing Gym unbounded-state warnings). Review found reader-release race and missed-heartbeat scheduling race; both reproduced with failing regressions and fixed. All motion tests used isolated ports/readers, not hardware. A successful SDK send is command acceptance, not target-reached acknowledgement.
