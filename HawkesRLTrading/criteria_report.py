#!/usr/bin/env python3
"""Score training runs against the three criteria, from the live metrics files.

Reads every episode_metrics_<label>.json under the given roots and reports:

  Criterion 1 -- front-running. Two things are reported, and the baseline
    matters as much as the contrast:

    * Response vs the PRE-TWAP window. `inventory_without_twap` pools the
      pre-TWAP stretch (100,250] with the post-TWAP stretch [400,550], and the
      pre stretch is exactly where front-running happens -- so a policy that
      builds its position before the meta-order starts and holds it through the
      window scores a paired shift of ~0 against that pooled baseline, BY
      CONSTRUCTION. The correct baseline is the pre window alone.

    * The SIDE CONTRAST:

      mean(inventory level in window | TWAP buying)
    - mean(inventory level in window | TWAP selling)

    which must be POSITIVE and large. Inventory level on a single-side run
    cannot establish front-running on its own, because a run can carry a
    persistent directional inventory bias that has nothing to do with the
    meta-order -- and the 2026-09-07 alternating-side runs do exactly that:
    both_base sits near +12 on buy AND +13 on sell episodes, both_onoff near
    -8.5 on buy AND -7 on sell. A side-independent bias reproduces "correct"
    single-side levels on whichever side happens to match its sign, which is
    how buy runs and sell runs from different arms could both look right.
    The contrast is immune to that bias because it differences it out.

    Level and the paired shift (in-window minus out-of-window) are both still
    reported, per side, as secondary.

  Criterion 2 -- profitable standalone. Mean terminal PnL over TWAP-ABSENT
    episodes only. Episodes where the meta-order is present are excluded
    entirely rather than split by window, because PnL is a path quantity and
    the out-of-window part of a present episode is still downstream of the
    meta-order's impact.

  Criterion 3 -- raises the TWAP's costs. Mean slippage in bps, per side,
    against a TWAP-alone control run (--control) that must have been produced
    with RL_DISABLED=true on the same seeds and schedule. Reported as the
    paired per-episode difference where episode indices line up, which removes
    the between-seed variance that dominates the unpaired comparison.

Usage:
    python3 HawkesRLTrading/criteria_report.py <dir> [<dir> ...] [--control <dir>]
    python3 HawkesRLTrading/criteria_report.py <dir> --last 20
"""
import argparse
import glob
import json
import math
import os


def _mean(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else None


def _stderr(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var / len(xs))


def _fmt(m, se, width=9):
    if m is None:
        return " " * (width - 3) + "n/a"
    s = "%*.2f" % (width, m)
    return s + (" +-%.2f" % (1.96 * se) if se is not None else "")


def load(path):
    with open(path) as fh:
        d = json.load(fh)
    return d


def find(roots):
    out = []
    for r in roots:
        if os.path.isfile(r):
            out.append(r)
            continue
        out.extend(sorted(glob.glob(os.path.join(r, "**", "episode_metrics_*.json"),
                                    recursive=True)))
    return out


def report(d, last=None, control=None):
    eps = d["episodes"]
    if last:
        eps = eps[-last:]
    present = [e for e in eps if e["twap_present"]]
    absent = [e for e in eps if not e["twap_present"]]
    buy = [e for e in present if e["side"] == "buy"]
    sell = [e for e in present if e["side"] == "sell"]

    cfg = ("bonus=%s explo=%s inv_pen=%s space=%s sym_mo=%s expApprox=%s%s"
           % (d.get("action_bonus"), d.get("exploration_bonus"),
              d.get("running_invpenalty"), d.get("action_space_config"),
              d.get("symmetric_mo_gating"), d.get("expApprox"),
              " RL_DISABLED" if d.get("rl_disabled") else ""))
    print("\n" + "=" * 78)
    print("%s   (%d episodes%s)" % (d["label"], len(eps),
                                    ", last %d" % last if last else ""))
    print("  " + cfg)
    print("  TWAP %s on / %s off, side=%s, seeds %s+%s"
          % (d.get("twap_on"), d.get("twap_off"), d.get("twap_side_mode"),
             d.get("seed_base"), d.get("seed_mode")))
    print("-" * 78)

    # --- criterion 1
    lb = [e["inv_level_in_window"] for e in buy]
    ls = [e["inv_level_in_window"] for e in sell]
    mb, ms = _mean(lb), _mean(ls)
    print("C1 front-running (want the CONTRAST positive and large)")
    print("     level | buy  n=%-3d %s" % (len(buy), _fmt(mb, _stderr(lb))))
    print("     level | sell n=%-3d %s" % (len(sell), _fmt(ms, _stderr(ls))))
    if mb is not None and ms is not None:
        contrast = mb - ms
        se = None
        if _stderr(lb) is not None and _stderr(ls) is not None:
            se = math.sqrt(_stderr(lb) ** 2 + _stderr(ls) ** 2)
        verdict = "PASS" if (contrast > 0 and se is not None and contrast > 1.96 * se) else "fail"
        print("     >>> SIDE CONTRAST  %s   %s" % (_fmt(contrast, se), verdict))
    elif mb is not None or ms is not None:
        print("     >>> SIDE CONTRAST  n/a  (single-side run -- scored by the")
        print("         PRESENT-ABSENT contrast below instead)")

    # PRESENT-ABSENT contrast. The only front-running measure available to a
    # single-side run. Compares the same clock window (twap_start, twap_end) on
    # episodes where the meta-order is there against episodes where it is not.
    # A per-run directional inventory bias sits in BOTH arms, so it differences
    # out -- the same cancellation that makes the side contrast work, but
    # across the presence axis instead of the side axis, so it does not need
    # both sides to exist. This is the measure the paper reports (+15.1 with
    # the meta-order against +10.9 without, for the buy-trained agent).
    wp = [e["inv_level_window_clock"] for e in present
          if e.get("inv_level_window_clock") is not None]
    wa = [e["inv_level_window_clock"] for e in absent
          if e.get("inv_level_window_clock") is not None]
    if wp and wa:
        # Sign of the target depends on the side the meta-order works.
        # Buying meta-order -> the agent should be MORE long when it is there.
        # Selling meta-order -> MORE short. A run carrying both sides has no
        # single target, so report per side.
        print("     present-absent contrast (front-running, single-side runs):")
        for nm, grp in (("buy", buy), ("sell", sell)):
            gp = [e["inv_level_window_clock"] for e in grp
                  if e.get("inv_level_window_clock") is not None]
            if not gp:
                continue
            c = _mean(gp) - _mean(wa)
            se = None
            if _stderr(gp) is not None and _stderr(wa) is not None:
                se = math.sqrt(_stderr(gp) ** 2 + _stderr(wa) ** 2)
            want = "> 0" if nm == "buy" else "< 0"
            ok = (c > 0) if nm == "buy" else (c < 0)
            v = "PASS" if (ok and se is not None and abs(c) > 1.96 * se) else "fail"
            print("       %-4s n=%-3d vs absent n=%-3d %s   want %s   %s"
                  % (nm, len(gp), len(wa), _fmt(c, se), want, v))
    elif wp and not wa:
        print("     present-absent contrast: n/a (no TWAP-absent episodes;")
        print("       set TWAP_OFF>0 to make criterion 1 readable on a single-side run)")
    elif not wp and (buy or sell):
        print("     present-absent contrast: n/a (run predates")
        print("       inv_level_window_clock -- absent episodes recorded no window level)")
    # Response against the PRE-TWAP baseline. The pooled out-of-window baseline
    # mixes (100,250] with [400,550], and the pre stretch is exactly where
    # front-running happens -- subtracting it subtracts the signal, so a policy
    # that builds before t=250 and holds scores ~0 by construction.
    def _resp(grp, key):
        return [e["inv_level_in_window"] - e[key]
                for e in grp if e.get("inv_level_in_window") is not None
                and e.get(key) is not None]
    rb, rs = _resp(buy, "inv_level_pre_window"), _resp(sell, "inv_level_pre_window")
    if rb or rs:
        print("     response vs PRE window (the front-running measure):")
        print("       buy  %s   want > 0      sell %s   want < 0"
              % (_fmt(_mean(rb), _stderr(rb)), _fmt(_mean(rs), _stderr(rs))))
        if rb and rs:
            c, se = _mean(rb) - _mean(rs), None
            if _stderr(rb) is not None and _stderr(rs) is not None:
                se = math.sqrt(_stderr(rb) ** 2 + _stderr(rs) ** 2)
            v = "PASS" if (c > 0 and se is not None and c > 1.96 * se) else "fail"
            print("       >>> RESPONSE CONTRAST %s   %s" % (_fmt(c, se), v))
    else:
        print("     response vs PRE window: n/a (run predates inv_level_pre_window;")
        print("       recover it post-hoc from inventorydists_* -- out[:len(inw)] is")
        print("       the pre stretch, validate with len(out) ~ 2*len(inw))")
    # ---- the user's stated goal, measured three ways ----------------------
    # "go long when TWAP is long and short when TWAP is short, IN GENERAL, not
    # for a specific window" -- so the headline for a single-side agent is the
    # WHOLE-EPISODE sign, not a window contrast. A standing directional
    # position is the target here, not an artefact.
    #
    # That alone is cheap: any constant bias scores on whichever side matches
    # its sign. What makes it non-trivial is that criterion 3 must ALSO pass on
    # the same run -- a bias unrelated to the meta-order does not raise the
    # meta-order's cost. Read C1-whole and C3 together; neither is sufficient.
    #
    # BEFORE is reported separately because the agent cannot condition on the
    # meta-order there: TWAPPresent is pinned to 0 outside [250,400]
    # (AR_RL_Trainer.py:668-672), so pre-window state is identical on buy, sell
    # and absent episodes. A correct sign BEFORE is therefore a prior baked
    # into the weights, not detection. It still counts toward the goal above;
    # it just is not evidence of anticipation.
    def _whole(e):
        nin, nout = e.get("n_in_window") or 0, e.get("n_out_window") or 0
        vin, vout = e.get("inv_level_in_window"), e.get("inv_level_out_window")
        tot = (nin if vin is not None else 0) + (nout if vout is not None else 0)
        if not tot:
            return None
        acc = (vin * nin if vin is not None else 0.0) + (vout * nout if vout is not None else 0.0)
        return acc / tot

    print("     the goal, three ways (buy wants > 0, sell wants < 0):")
    for nm, grp in (("buy", buy), ("sell", sell)):
        if not grp:
            continue
        rows = []
        for tag, vals in (("WHOLE-EPISODE", [_whole(e) for e in grp]),
                          ("BEFORE (pre)", [e.get("inv_level_pre_window") for e in grp]),
                          ("IN-WINDOW", [e.get("inv_level_in_window") for e in grp])):
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            m, se = _mean(vals), _stderr(vals)
            ok = (m > 0) if nm == "buy" else (m < 0)
            v = "PASS" if (ok and se is not None and abs(m) > 1.96 * se) else "fail"
            rows.append("       %-4s %-13s n=%-3d %s   %s" % (nm, tag, len(vals), _fmt(m, se), v))
        print("\n".join(rows))

    print("     shift vs POOLED out-window (legacy, broken by construction):")
    print("       buy  %s      sell %s"
          % (_fmt(_mean(_resp(buy, "inv_level_out_window")), None),
             _fmt(_mean(_resp(sell, "inv_level_out_window")), None)))

    # --- criterion 2
    pa = [e["terminal_pnl"] for e in absent]
    pp = [e["terminal_pnl"] for e in present]
    print("C2 standalone PnL (want > 0, on TWAP-ABSENT episodes)")
    if absent:
        m = _mean(pa)
        verdict = "PASS" if (m is not None and m > 0 and _stderr(pa) is not None
                             and m > 1.96 * _stderr(pa)) else "fail"
        print("     absent n=%-3d %s   %s" % (len(absent), _fmt(m, _stderr(pa)), verdict))
    else:
        print("     absent n=0    n/a  (this run has no TWAP-off episodes;")
        print("       set TWAP_OFF>0 -- criterion 2 cannot be read from present episodes)")
    print("     present n=%-3d %s  (context only)" % (len(pp), _fmt(_mean(pp), _stderr(pp))))

    # --- criterion 3
    print("C3 TWAP slippage in bps (want ABOVE the TWAP-alone control)")
    for name, grp in (("buy", buy), ("sell", sell)):
        sl = [e["twap_slippage_bps"] for e in grp]
        line = "     %-4s n=%-3d %s" % (name, len(grp), _fmt(_mean(sl), _stderr(sl)))
        if control is not None:
            # Pair on SEED, not episode index. Episode index only lines up if
            # both runs used the same seed base, side mode and presence cycle;
            # pairing on it blindly would silently compare different market
            # paths and call the difference an effect of the agent.
            cmap = {e.get("seed"): e["twap_slippage_bps"] for e in control["episodes"]
                    if e["twap_present"] and e["side"] == name and e.get("seed") is not None}
            pairs = [(e["twap_slippage_bps"], cmap[e.get("seed")]) for e in grp
                     if e.get("seed") in cmap and e["twap_slippage_bps"] is not None
                     and cmap[e.get("seed")] is not None]
            if pairs:
                diffs = [a - b for a, b in pairs]
                m, se = _mean(diffs), _stderr(diffs)
                verdict = "PASS" if (m is not None and se is not None
                                     and m > 1.96 * se) else "fail"
                line += "   paired vs control n=%-3d %s  %s" % (
                    len(diffs), _fmt(m, se), verdict)
            else:
                line += "   (no paired control episodes)"
        print(line)
    if control is not None:
        mismatch = [k for k in ("twap_on", "twap_off", "twap_side_mode",
                                "seed_base", "seed_mode", "expApprox")
                    if control.get(k) != d.get(k)]
        if mismatch:
            print("     !! control differs from this run in: %s" % ", ".join(mismatch))
            print("        seeds/schedule must match for the pairing to mean anything")
    ex = [e["twap_total_executed"] for e in present]
    print("     twap executed (order=150): %s" % _fmt(_mean(ex), None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--last", type=int, default=None,
                    help="use only the last N episodes (policy at the end of training)")
    ap.add_argument("--control", default=None,
                    help="dir or json of a TWAP-alone run (RL_DISABLED=true) for criterion 3")
    a = ap.parse_args()

    control = None
    if a.control:
        paths = find([a.control])
        if not paths:
            raise SystemExit("no episode_metrics_*.json under %s" % a.control)
        if len(paths) > 1:
            raise SystemExit("--control matched %d metrics files; name one:\n  %s"
                             % (len(paths), "\n  ".join(paths)))
        control = load(paths[0])
        if not control.get("rl_disabled"):
            raise SystemExit("control run %s was NOT produced with RL_DISABLED=true; "
                             "it is not a TWAP-alone baseline" % paths[0])

    paths = find(a.roots)
    if not paths:
        raise SystemExit("no episode_metrics_*.json found")
    for p in paths:
        report(load(p), last=a.last, control=control)
    print()


if __name__ == "__main__":
    main()
