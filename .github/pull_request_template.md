## What this changes

<!-- What behaves differently now, and why. Link the issue if there is one: "Fixes #123". -->

## How it was verified

<!-- The commands you ran and what they reported. Name any tests that skipped and why. -->

## Checklist

- [ ] Tests added or updated, and they check real inputs and outputs (not a mock of the thing under test)
- [ ] The tests I touched pass locally (`python -m pytest <files>`), and the full suite passes or CI is green
- [ ] `docs/features/registry.py` updated if a capability was added or changed; `python tools/feature_status.py --strict` shows no downgrade
- [ ] `docs/FEATURES.md` was not edited by hand
- [ ] No-fabrication rule respected: nothing synthesised, assumed or heuristic is presented as measured; refusals say why
- [ ] If a fabrication was removed, a regression test was added (e.g. in `tests/test_honesty.py`)
- [ ] Docs updated for any behaviour change (README, `docs/`), and `CHANGELOG.md` under Unreleased
- [ ] No secrets, `.env` files, model weights, datasets or run outputs committed
- [ ] For UI changes: before/after screenshots below

## Screenshots

<!-- For UI changes. Delete this section otherwise. -->
