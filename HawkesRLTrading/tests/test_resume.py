"""Pin checkpoint resume, in the trainer and in ModelManager.

Models are checkpointed every 4 episodes, but nothing could use them:

  * `checkpoint_params` was hardcoded to None, so a job could not ask to resume.
  * Even with weights loaded, the loop ran `range(N_EPISODES)` from 0 and the
    setup/load block was gated on `episode == 0`, so a resume redid every
    episode from scratch.
  * `load_models(timestamp=None)` sorted metadata filenames as STRINGS and took
    the first, so `_epoch_8_` beat `_epoch_76_` and it silently loaded epoch 8
    of an 80-episode run. (The explicit-timestamp path is fine: callers pass
    the timestamp with the label already appended.)
"""
import json
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

TRAINER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "AR_RL_Trainer.py")


def _src():
    return open(TRAINER).read()


# ---------------------------------------------------------------- ModelManager

class _Net:
    def state_dict(self):
        return {}

    def load_state_dict(self, sd):
        return None


class _Marked(_Net):
    """Records which checkpoint's weights were actually loaded into it."""
    seen = None

    def load_state_dict(self, sd):
        self.seen = float(sd["marker"][0])
        return None


def _populate(d, label, timestamp, epochs):
    import torch
    from HJBQVI.utils import ModelManager
    mm = ModelManager(model_dir=d, label=label)
    mm.timestamp = timestamp
    for e in epochs:
        mm.save_models(epoch=e, d=_Net(), u=_Net())
    return mm


def test_latest_checkpoint_is_chosen_numerically_not_lexicographically():
    """`_epoch_8_` sorts after `_epoch_76_` as a string. The old code took the
    string maximum and loaded epoch 8 of an 80-episode run."""
    d = tempfile.mkdtemp()
    try:
        mm = _populate(d, "dfx_sell_base", "20260907_085440", [0, 4, 8, 12, 64, 72, 76])
        names = [f for f in os.listdir(d) if f.startswith("model_metadata")]
        assert sorted(names, reverse=True)[0].startswith("model_metadata_epoch_8_"), \
            "fixture does not reproduce the lexicographic trap"
        # Behavioural: make each epoch's weights distinguishable, then check
        # WHICH ones came back. Re-deriving the expected filename in the test
        # would pass against a loader that ignores epochs entirely.
        import torch
        for e in (8, 76):
            meta = os.path.join(d, f"model_metadata_epoch_{e}_20260907_085440_dfx_sell_base.json")
            j = json.load(open(meta))
            for k in j["models"]:
                torch.save({"marker": torch.tensor([float(e)])}, j["models"][k])
        holder = _Marked()
        mm.load_models(timestamp=None, epoch=-1, d=holder, u=_Marked())
        assert holder.seen == 76.0, (
            "loaded epoch %s, expected 76 -- lexicographic sort puts _epoch_8_ first"
            % holder.seen)
    finally:
        shutil.rmtree(d)


def test_explicit_timestamp_carries_the_label_by_convention():
    """The caller passes the timestamp WITH the label already appended --
    AR_RL_runner.py:44 passes '20260630_214031_train_new_vf_explo_gae_sell'.
    load_models must NOT append self.label again; doing so yields a
    double-label path and breaks every existing call."""
    d = tempfile.mkdtemp()
    try:
        mm = _populate(d, "train_new_vf_explo_gae_sell", "20260630_214031", [76])
        out = mm.load_models(timestamp="20260630_214031_train_new_vf_explo_gae_sell",
                             epoch=76, d=_Net(), u=_Net())
        assert isinstance(out, dict) and set(out) == {"d", "u"}, out
        assert all(v is not None for v in out.values()), out
    finally:
        shutil.rmtree(d)


def test_another_runs_checkpoints_are_not_picked_up():
    """A shared model_dir must not leak another label's weights. Behavioural:
    the other run has a HIGHER epoch, so a loader that ignores labels would
    prefer it."""
    import torch
    d = tempfile.mkdtemp()
    try:
        _populate(d, "other_run", "20260907_999999", [90])
        mm = _populate(d, "mine", "20260907_085440", [4])
        for lbl, ep, mark in (("other_run", 90, 90.0), ("mine", 4, 4.0)):
            ts = "20260907_999999" if lbl == "other_run" else "20260907_085440"
            j = json.load(open(os.path.join(d, f"model_metadata_epoch_{ep}_{ts}_{lbl}.json")))
            for k in j["models"]:
                torch.save({"marker": torch.tensor([mark])}, j["models"][k])
        holder = _Marked()
        mm.load_models(timestamp=None, epoch=-1, d=holder, u=_Marked())
        assert holder.seen == 4.0, "loaded another label's weights (%s)" % holder.seen
    finally:
        shutil.rmtree(d)


def test_label_match_is_exact_not_a_suffix():
    """`endswith("_base.json")` also matches "..._sell_base.json"."""
    import torch
    d = tempfile.mkdtemp()
    try:
        _populate(d, "sell_base", "20260907_999999", [90])
        mm = _populate(d, "base", "20260907_085440", [4])
        for lbl, ep, ts, mark in (("sell_base", 90, "20260907_999999", 90.0),
                                  ("base", 4, "20260907_085440", 4.0)):
            j = json.load(open(os.path.join(d, f"model_metadata_epoch_{ep}_{ts}_{lbl}.json")))
            for k in j["models"]:
                torch.save({"marker": torch.tensor([mark])}, j["models"][k])
        holder = _Marked()
        mm.load_models(timestamp=None, epoch=-1, d=holder, u=_Marked())
        assert holder.seen == 4.0, "suffix match picked up sell_base (%s)" % holder.seen
    finally:
        shutil.rmtree(d)


def test_newest_run_wins_at_a_tied_epoch():
    """Two runs sharing a label and an epoch: os.listdir order must not decide."""
    import torch
    d = tempfile.mkdtemp()
    try:
        _populate(d, "lbl", "20260901_000000", [4])
        mm = _populate(d, "lbl", "20260913_120000", [4])
        for ts, mark in (("20260901_000000", 1.0), ("20260913_120000", 2.0)):
            j = json.load(open(os.path.join(d, f"model_metadata_epoch_4_{ts}_lbl.json")))
            for k in j["models"]:
                torch.save({"marker": torch.tensor([mark])}, j["models"][k])
        holder = _Marked()
        mm.load_models(timestamp=None, epoch=4, d=holder, u=_Marked())
        assert holder.seen == 2.0, "picked the older run (%s)" % holder.seen
    finally:
        shutil.rmtree(d)


def test_a_failed_load_returns_a_dict_not_a_list():
    """The trainer indexes loaded_models['d']; a list gave
    `TypeError: list indices must be integers` instead of a diagnosable error."""
    d = tempfile.mkdtemp()
    try:
        mm = _populate(d, "lbl", "20260907_085440", [4])
        out = mm.load_models(timestamp=None, epoch=99, d=_Net(), u=_Net())
        assert isinstance(out, dict), type(out)
        assert out["d"] is None and out["u"] is None, out
    finally:
        shutil.rmtree(d)


# -------------------------------------------------------------------- trainer

def test_checkpoint_params_is_env_driven():
    src = _src()
    assert "checkpoint_params = None if RESUME_EPOCH is None else (RESUME_TIMESTAMP, RESUME_EPOCH)" in src
    assert 'os.environ.get("RESUME_EPOCH")' in src


def test_loop_starts_at_start_episode():
    src = _src()
    assert "for episode in range(START_EPISODE, N_EPISODES):" in src, \
        "the loop still restarts at 0, so a resume redoes every episode"


def _start_episode(env, model_dir="/nonexistent", label="lbl"):
    """Run the trainer's START_EPISODE resolution under a given environment."""
    src = _src()
    blk = src[src.index("_RESUME_EPOCH_RAW = "):src.index("#the time that the TWAP")]
    ns = {"os": os, "model_dir": model_dir, "label": label}
    saved = {k: os.environ.get(k) for k in ("RESUME_EPOCH", "RESUME_TIMESTAMP", "START_EPISODE")}
    try:
        for k in saved:
            os.environ.pop(k, None)
        os.environ.update(env)
        exec(blk, ns)
        return ns["START_EPISODE"], ns["RESUME_EPOCH"]
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_no_resume_starts_at_zero():
    assert _start_episode({}) == (0, None)


def test_start_episode_defaults_to_the_checkpoint_plus_one():
    assert _start_episode({"RESUME_EPOCH": "40"})[0] == 41


def test_explicit_start_episode_wins():
    assert _start_episode({"RESUME_EPOCH": "40", "START_EPISODE": "12"})[0] == 12


def test_resume_epoch_minus_one_does_not_restart_at_zero():
    """-1 means "the newest checkpoint". `-1 + 1 == 0` would restart the loop
    at episode 0 and redo the entire run -- defeating the resume on the
    spelling a user is most likely to type."""
    import tempfile as _tf
    import shutil as _sh
    d = _tf.mkdtemp()
    try:
        _populate(d, "lbl", "20260907_085440", [0, 4, 8, 76])
        got, _ = _start_episode({"RESUME_EPOCH": "-1"}, model_dir=d, label="lbl")
        assert got == 77, got
    finally:
        _sh.rmtree(d)


def test_resume_epoch_minus_one_fails_loudly_with_no_checkpoints():
    """Silently starting at 0 would look like a fresh run and quietly discard
    whatever the user meant to resume from."""
    import tempfile as _tf
    import shutil as _sh
    d = _tf.mkdtemp()
    try:
        try:
            _start_episode({"RESUME_EPOCH": "-1"}, model_dir=d, label="lbl")
        except SystemExit:
            return
        raise AssertionError("expected SystemExit when no checkpoint exists")
    finally:
        _sh.rmtree(d)


def test_network_setup_is_not_gated_on_episode_zero():
    """`if episode == 0:` never fires on a resume, so the networks would never
    be built and the checkpoint never loaded."""
    src = _src()
    assert "if episode == START_EPISODE:" in src
    assert "if episode == 0:" not in src


def test_side_counter_accounts_for_skipped_episodes():
    """Left at 0, `alternate` would restart on buy after a resume and the
    buy/sell balance would be wrong for the remainder of the run."""
    src = _src()
    assert "n_twap_present = sum(1 for _e in range(START_EPISODE)" in src


def test_resumed_side_matches_what_an_uninterrupted_run_would_have_used():
    """Not the counter -- the SIDE. Simulate the trainer's alternation for a
    full run, then for a run resumed part-way, and require the side of every
    post-resume episode to agree."""
    def sides(on, off, mode, lo, hi, n_start):
        n_present, out = n_start, {}
        for e in range(lo, hi):
            present = (e % (on + off)) < on
            side = ("buy" if n_present % 2 == 0 else "sell") if mode == "alternate" else mode
            if present:
                n_present += 1
            out[e] = side if present else "none"
        return out

    for on, off in ((1, 0), (2, 1), (3, 2)):
        full = sides(on, off, "alternate", 0, 120, 0)
        for start in (1, 5, 40, 77):
            seeded = sum(1 for e in range(start) if (e % (on + off)) < on)
            resumed = sides(on, off, "alternate", start, 120, seeded)
            assert all(resumed[e] == full[e] for e in resumed), (on, off, start)
            # and prove the test can fail: starting the counter at 0 must break it
            naive = sides(on, off, "alternate", start, 120, 0)
            if seeded % 2:
                assert any(naive[e] != full[e] for e in naive), (on, off, start)


def test_prior_episode_metrics_are_kept():
    src = _src()
    assert "if START_EPISODE > 0:" in src
    assert 'e.get("episode", 0) < START_EPISODE' in src, \
        "resume must keep earlier episodes and drop any it is about to redo"


def test_resume_is_reported_in_the_banner():
    src = _src()
    assert "{START_EPISODE}" in src and "{checkpoint_params}" in src


if __name__ == "__main__":
    mod = sys.modules[__name__]
    failures = 0
    for name in sorted(dir(mod)):
        if not name.startswith("test_"):
            continue
        try:
            getattr(mod, name)()
            print("PASS %s" % name)
        except AssertionError as e:
            failures += 1
            print("FAIL %s: %s" % (name, e))
        except Exception as e:
            failures += 1
            print("ERROR %s: %r" % (name, e))
    print("all tests passed" if not failures else "%d FAILURES" % failures)
    sys.exit(1 if failures else 0)
