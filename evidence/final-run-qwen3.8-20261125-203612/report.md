# Session 20261125-203612-advance-expiry

Task: Optional expiry date for customer advances (advance-expiry)
Project: openstock
Starting commit: `abc87dccab96c78917b35add542284e61adc5f37`
Model: qwen3.8

**State:** completed

**Status:** PASSED: agent finished and all required checks passed

Agent summary: Added optional expiry date for customer advances: new migration migrations/011_advance_expiry.sql adds expires_on column to payments; DB.register_advance validates optional expires_on (ValueError mentioning 'expiry date' for bad or past dates) and stores it; get_advance_balance and use_advance skip expired advances; new DB.release_expired_advances() zeroes remaining on expired advances, inserts 'release' rows with the customer's company_id and negative amount, returns total released (idempotent); get_advances rows include expires_on; app.py POST /api/advance passes optional expires_on through, returning 400 on invalid values. All required checks pass.

## Counters
- actions: 27
- invalid requests: 0
- denied actions: 0
- model retries: 1
- repair rounds: 0
- shortened outputs: 1

## Checks
- baseline acceptance on starting code: failed (exit 1)
- acceptance: passed (exit 0)
- regression_core: passed (exit 0)
- regression_reversal: passed (exit 0)
- build: passed (exit 0)

## Changed files
- app.py
- db.py
- migrations/011_advance_expiry.sql

## Unfinished work
(nothing)

Session folder: `C:\Users\Patrick\Projects\AAASD-stage-2-coding-harness\runs\20261125-203612-advance-expiry`
