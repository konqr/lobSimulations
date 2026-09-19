"""Pin the arm-parameter env overrides in AR_RL_Trainer.py.

All six of the 2026-09-07 runs launched on the wrong arm (exploration_bonus=0
when the plan called for 0.1) because the arm was hardcoded and nothing in the
job recorded it. The overrides exist so the arm travels with the submission
script. The failure mode worth pinning is EXP_APPROX's string->bool parse:
`bool("False")` is True, so a naive conversion would silently flip the whole
simulator regime for anyone who wrote EXP_APPROX=False.
"""
import os
import re
import sys

TRAINER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "AR_RL_Trainer.py")

_NAMES = ("EXPLORATION_BONUS", "GAE_LAMBDA", "EXP_APPROX",
          "ACTION_BONUS", "RUNNING_INVPENALTY", "ENTROPY_COEF",
          "ACTION_SPACE_CONFIG", "SYMMETRIC_MO_GATING", "RL_DISABLED",
          "TERMINAL_INVPENALTY", "FIRST_VISIT_BONUS")


def _config_lines():
    """The three assignment lines, pulled verbatim out of the trainer source."""
    src = open(TRAINER).read()
    lines = {}
    for name in _NAMES:
        m = re.search(r"^%s = .*$" % name, src, re.M)
        assert m is not None, "%s assignment not found in AR_RL_Trainer.py" % name
        lines[name] = m.group(0)
    return lines


def _evaluate(env):
    """Exec the assignment lines under a given environment, return the values."""
    lines = _config_lines()
    saved = {k: os.environ.get(k) for k in _NAMES}
    try:
        for k in _NAMES:
            os.environ.pop(k, None)
        os.environ.update(env)
        ns = {"os": os}
        for name in _NAMES:
            exec(lines[name], ns)
        return {name: ns[name] for name in _NAMES}
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_defaults_are_the_combined_arm():
    """Unset env must give the explo_gae arm -- the one the plan asks for."""
    v = _evaluate({})
    assert v["EXPLORATION_BONUS"] == 0.1, v
    assert v["GAE_LAMBDA"] == 0.95, v
    assert v["EXP_APPROX"] is False, v


def test_every_other_default_reproduces_historical_behaviour():
    """Only the arm defaults above are allowed to differ from the six runs on
    the cluster. Everything else must be bit-identical to them, or new runs
    cannot be compared with old ones."""
    v = _evaluate({})
    assert v["ACTION_BONUS"] == 0.5, v
    assert v["RUNNING_INVPENALTY"] == 0.0, v
    assert v["ENTROPY_COEF"] == 0.0, v
    assert v["ACTION_SPACE_CONFIG"] == 1, v
    assert v["SYMMETRIC_MO_GATING"] is False, v
    assert v["RL_DISABLED"] is False, v
    assert v["TERMINAL_INVPENALTY"] == 25.0, v   # 5 * eta
    assert v["FIRST_VISIT_BONUS"] == 0.2, v


def test_boolean_arms_reject_the_naive_parse():
    """bool("False") is True for SYMMETRIC_MO_GATING and RL_DISABLED too."""
    for name in ("SYMMETRIC_MO_GATING", "RL_DISABLED"):
        for spelling in ("false", "False", "0", "no", ""):
            assert _evaluate({name: spelling})[name] is False, (name, spelling)
        for spelling in ("true", "True", "1", "yes"):
            assert _evaluate({name: spelling})[name] is True, (name, spelling)


def test_control_arm_skips_the_policy_call_not_just_its_result():
    """get_action draws from the stdlib RNG the exchange also uses. Calling it
    and discarding the answer would desynchronise the background order flow and
    make the control a different market, which would invalidate criterion 3."""
    src = open(TRAINER).read()
    assert "if RL_DISABLED:" in src
    i = src.index("if RL_DISABLED:")
    # the control branch runs up to its matching `else:` at the same indent
    j = src.index("\n                else:", i)
    block = src[i:j]
    assert "get_action" not in block, \
        "the control arm must not call get_action at all"
    assert "(agent.id, (12, 1))" in block, "control arm must submit the no-op"
    # and the treatment branch on the other side of the else must still call it
    assert "agent.get_action" in src[j:j + 400]


def test_control_arm_does_not_train():
    src = open(TRAINER).read()
    assert "if (not RL_DISABLED) and ('test' not in label)" in src


def test_reporting_survives_an_empty_trajectory_buffer():
    """The control arm stores no transitions, so every per-episode reporting
    path that indexes the buffer must be guarded. An unguarded
    `agent.trajectory_buffer[0][0]` crashed the first control run at the end of
    its first episode."""
    src = open(TRAINER).read()
    i = src.index("episodic_rewards = []")
    block = src[i:i + 900]
    assert "if len(agent.trajectory_buffer) > 0:" in block, \
        "trajectory_buffer is indexed without a length check"
    j = block.index("if len(agent.trajectory_buffer) > 0:")
    k = block.index("agent.trajectory_buffer[0][0]")
    assert j < k, "the length check must precede the indexing"
    assert "if episodic_rewards else np.nan" in block, \
        "np.mean/np.std over an empty episodic_rewards list"


def test_gae_lambda_only_arm_is_reachable():
    v = _evaluate({"EXPLORATION_BONUS": "0"})
    assert v["EXPLORATION_BONUS"] == 0.0, v
    assert v["GAE_LAMBDA"] == 0.95, v


def test_exp_approx_truthy_spellings():
    for spelling in ("1", "true", "True", "TRUE", "yes", " true "):
        v = _evaluate({"EXP_APPROX": spelling})
        assert v["EXP_APPROX"] is True, (spelling, v)


def test_exp_approx_falsy_spellings_are_not_naively_truthy():
    """bool("False") is True. A naive parse here silently changes TAU 500->10."""
    for spelling in ("0", "false", "False", "FALSE", "no", ""):
        v = _evaluate({"EXP_APPROX": spelling})
        assert v["EXP_APPROX"] is False, (spelling, v)


def test_arm_params_are_actually_wired_into_the_agent():
    """An override that never reaches the constructor is worse than no override."""
    src = open(TRAINER).read()
    assert "exploration_bonus = EXPLORATION_BONUS" in src
    assert "gae_lambda=GAE_LAMBDA" in src
    assert "terminal_invpenalty=TERMINAL_INVPENALTY" in src
    assert "first_visit_bonus=FIRST_VISIT_BONUS" in src
    assert "'expApprox' : EXP_APPROX" in src
    # and no stale hardcoded copy left behind
    assert "exploration_bonus = 0," not in src
    assert "gae_lambda=0.95" not in src
    assert "terminal_invpenalty=5*eta" not in src
    assert "'expApprox' : False" not in src


def test_config_banner_reports_the_arm():
    """The .o file must state the arm, or runs are indistinguishable after the fact."""
    src = open(TRAINER).read()
    for name in _NAMES:
        assert ("{%s}" % name) in src, "%s missing from the config banner" % name


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
    print("all tests passed" if not failures else "%d FAILURES" % failures)
    sys.exit(1 if failures else 0)
