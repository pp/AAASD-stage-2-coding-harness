"""Feature acceptance check: customer advances with an optional expiry date.

Kept outside the agent's writable area (mounted read-only in the sandbox).
Seeds its own database in a temporary folder, so the agent cannot change the data it is judged on.
Run inside the sandbox with REPO_DIR pointing at the repository (default /work).
Exit code 0 = pass, 1 = fail, 2 = setup problem.

Each check runs on its own, so one failure does not hide the others.
"""
import os
import subprocess
import sys
import tempfile
import traceback

REPO = os.environ.get("REPO_DIR", "/work")
sys.path.insert(0, REPO)

workdir = tempfile.mkdtemp(prefix="acceptance-")
os.environ["OPENSTOCK_DB"] = os.path.join(workdir, "acceptance.db")
seed = subprocess.run(
    [sys.executable, os.path.join(REPO, "seed_demo.py"), "--password", "acceptance-only-password"],
    cwd=REPO, capture_output=True, text=True,
)
if seed.returncode != 0:
    print("SETUP FAILED: seed_demo.py did not run")
    print(seed.stdout)
    print(seed.stderr)
    sys.exit(2)

from db import DB, load_config  # noqa: E402

db = DB(load_config())
db.connect()
db.apply_migrations()
cur = db.cursor()

cur.execute("SELECT code FROM companies WHERE code IS NOT NULL ORDER BY code LIMIT 2")
rows = cur.fetchall()
CUSTOMER = rows[0][0]
OTHER = rows[1][0]

FUTURE = "2099-12-31"
PAST = "2000-01-01"

results = []


def record(name, ok, detail=""):
    results.append((name, ok, detail))
    print("{} {}{}".format("PASS" if ok else "FAIL", name, (" - " + detail) if detail and not ok else ""))


def balance(company):
    return float(db.get_advance_balance(company))


def advance_rows(company):
    cur.execute("SELECT COUNT(*) FROM payments WHERE company_id=? AND type='advance'", (company,))
    return cur.fetchone()[0]


def row_id_for(amount, company):
    cur.execute("SELECT id FROM payments WHERE company_id=? AND type='advance' AND amount=? "
                "ORDER BY id DESC LIMIT 1", (company, amount))
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("no advance of {} was stored for this test (see valid_expiry_is_stored_and_listed)".format(amount))
    return row[0]


def case(fn):
    def wrapper():
        try:
            fn()
        except Exception as exc:  # reported as a failure, the other checks still run
            record(fn.__name__, False, "{}: {}".format(type(exc).__name__, exc))
            if os.environ.get("ACCEPTANCE_DEBUG"):
                traceback.print_exc()
    wrapper.__name__ = fn.__name__
    return wrapper


@case
def advance_without_expiry_still_works():
    before = balance(CUSTOMER)
    db.register_advance(CUSTOMER, 100, operator="ACC")
    after = balance(CUSTOMER)
    ok = abs((after - before) - 100) < 1e-6
    record("advance_without_expiry_still_works", ok, "balance went from {} to {}".format(before, after))


@case
def invalid_expiry_text_is_rejected_and_stores_nothing():
    before_rows, before_balance = advance_rows(CUSTOMER), balance(CUSTOMER)
    try:
        db.register_advance(CUSTOMER, 50, operator="ACC", expires_on="31/12/2099")
    except ValueError as exc:
        ok = "expiry" in str(exc).lower()
        msg = "rejected with: {}".format(exc)
        if not ok:
            msg += " (message does not mention the expiry date)"
    else:
        ok, msg = False, "register_advance accepted an invalid expiry date"
    unchanged = advance_rows(CUSTOMER) == before_rows and abs(balance(CUSTOMER) - before_balance) < 1e-6
    record("invalid_expiry_text_is_rejected_and_stores_nothing", ok and unchanged,
           msg + ("" if unchanged else "; a row or balance changed"))


@case
def expiry_in_the_past_is_rejected_at_registration():
    before_rows = advance_rows(CUSTOMER)
    try:
        db.register_advance(CUSTOMER, 50, operator="ACC", expires_on=PAST)
    except ValueError as exc:
        ok, msg = advance_rows(CUSTOMER) == before_rows, "rejected with: {}".format(exc)
    else:
        ok, msg = False, "an advance that is already expired was accepted"
    record("expiry_in_the_past_is_rejected_at_registration", ok, msg)


@case
def valid_expiry_is_stored_and_listed():
    db.register_advance(CUSTOMER, 70, operator="ACC", expires_on=FUTURE)
    rid = row_id_for(70, CUSTOMER)
    cur.execute("SELECT expires_on FROM payments WHERE id=?", (rid,))
    stored = cur.fetchone()[0]
    listed = [a for a in db.get_advances(company_id=CUSTOMER) if a["id"] == rid]
    ok = stored == FUTURE and bool(listed) and listed[0].get("expires_on") == FUTURE
    record("valid_expiry_is_stored_and_listed", ok,
           "stored={!r}, listed={}".format(stored, listed[0].get("expires_on") if listed else None))


@case
def expired_advance_is_not_counted_or_used():
    rid = row_id_for(70, CUSTOMER)
    cur.execute("UPDATE payments SET expires_on=? WHERE id=?", (PAST, rid))
    db.conn.commit()
    before = balance(CUSTOMER)
    # The 70 advance is expired, so the balance must not include it.
    # Expected balance = all other advances left, which is at least 100 from the first check.
    used = db.use_advance(CUSTOMER, 1000, doc_no="ACC-2")
    cur.execute("SELECT remaining FROM payments WHERE id=?", (rid,))
    still_there = float(cur.fetchone()[0])
    ok = (abs(still_there - 70) < 1e-6 and used <= before + 1e-6 and abs(balance(CUSTOMER) - (before - used)) < 1e-6)
    record("expired_advance_is_not_counted_or_used", ok,
           "balance before={}, used={}, expired row remaining={}".format(before, used, still_there))


@case
def release_frees_expired_balance_once():
    rid = row_id_for(70, CUSTOMER)
    released = float(db.release_expired_advances())
    cur.execute("SELECT remaining FROM payments WHERE id=?", (rid,))
    remaining = float(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM payments WHERE type='release'")
    release_rows = cur.fetchone()[0]
    second = float(db.release_expired_advances())
    ok = abs(released - 70) < 1e-6 and abs(remaining) < 1e-6 and release_rows >= 1 and abs(second) < 1e-6
    record("release_frees_expired_balance_once", ok,
           "released={}, remaining={}, release rows={}, second call={}".format(
               released, remaining, release_rows, second))


@case
def release_row_keeps_the_customer():
    # Ledger rule: a release row must name the customer whose advance expired.
    db.register_advance(CUSTOMER, 45, operator="ACC", expires_on=FUTURE)
    rid = row_id_for(45, CUSTOMER)
    cur.execute("UPDATE payments SET expires_on=? WHERE id=?", (PAST, rid))
    db.conn.commit()
    db.release_expired_advances()
    cur.execute("SELECT company_id FROM payments WHERE type='release' AND amount=-45 ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    ok = row is not None and row[0] == CUSTOMER
    record("release_row_keeps_the_customer", ok,
           "release row company_id={!r}, expected {!r}".format(row[0] if row else None, CUSTOMER))


@case
def unexpired_advance_is_still_usable_after_release():
    db.register_advance(OTHER, 30, operator="ACC", expires_on=FUTURE)
    used = db.use_advance(OTHER, 30, doc_no="ACC-3")
    ok = abs(used - 30) < 1e-6
    record("unexpired_advance_is_still_usable_after_release", ok, "used={}".format(used))


@case
def expiry_stays_optional_for_other_customers():
    db.register_advance(OTHER, 20, operator="ACC")
    used = db.use_advance(OTHER, 20, doc_no="ACC-4")
    record("expiry_stays_optional_for_other_customers", abs(used - 20) < 1e-6, "used={}".format(used))


def main():
    for fn in (advance_without_expiry_still_works, invalid_expiry_text_is_rejected_and_stores_nothing,
               expiry_in_the_past_is_rejected_at_registration, valid_expiry_is_stored_and_listed,
               expired_advance_is_not_counted_or_used, release_frees_expired_balance_once,
               unexpired_advance_is_still_usable_after_release, release_row_keeps_the_customer,
               expiry_stays_optional_for_other_customers):
        fn()
    failures = [name for name, ok, _ in results if not ok]
    print()
    if failures:
        print("ACCEPTANCE FAILED ({} of {} checks): {}".format(len(failures), len(results), ", ".join(failures)))
        return 1
    print("ACCEPTANCE PASSED ({} checks)".format(len(results)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
