#!/usr/bin/env python3
# Exact Score Predictor — Priors + EWMA + FPL availability + Fixture Difficulty
# Uses LOCAL-MAP score selection (3x3 around expected goals) to avoid 2–1 clustering.
# Keeps aliases; promoted handling; gentle tilts; tiny rho; no total-goals cap.
# Adds: Result (H/D/A), Result %, Market Odds editing, and a "Check Data" button.
# NEW: TILT_CAP to prevent extreme scorelines like 6–0 in lopsided matches.

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading, requests, re, json, math, time
from collections import defaultdict

# =========================
# Tunables
# =========================
DEF_EXP        = 0.90    # Understat xGA -> defence softness exponent
RHO            = 0.08    # tiny correlation: allows low scores without draw clustering
MAX_GOALS      = 6

# Non-sum-preserving tilts (allow favourites to pull away, but gently)
HOME_ADV       = 1.03    # modest home lift
K_TILT         = 0.80    # strength-ratio exponent (higher => more asymmetry)
TILT_CAP       = 1.60    # NEW: cap the tilt multiplier to avoid extreme separations

# Player availability layer (FPL)
USE_PLAYER_LAYER   = True
ATK_TOP_N          = 5
DEF_TOP_N          = 4
ATK_CLIP           = (0.80, 1.25)
DEF_CLIP           = (0.85, 1.20)
DEF_AVAIL_EXP      = 1.00

# Promoted-team adjustment (if missing last-season EPL priors)
PROM_ATK           = 0.90     # trim their attack baseline
PROM_DEF           = 1.10     # worsen their conceded side

# Betting utilities
DEFAULT_EXCH_COMM  = 0.02     # 2% exchange commission for fair-odds adj

# =========================
# Team name normalisation (user input, FPL, Understat)
# =========================
SLUG_ALIASES = {
    # Big 6
    "arsenal":"arsenal",
    "manchester city":"man_city","man city":"man_city","man c":"man_city",
    "manchester united":"man_utd","man utd":"man_utd","man united":"man_utd","man u":"man_utd",
    "liverpool":"liverpool",
    "chelsea":"chelsea",
    "tottenham":"spurs","tottenham hotspur":"spurs","spurs":"spurs",
    # Others in current EPL list
    "aston villa":"aston_villa","villa":"aston_villa",
    "brighton":"brighton","brighton & hove albion":"brighton","brighton and hove albion":"brighton",
    "bournemouth":"bournemouth","afc bournemouth":"bournemouth",
    "brentford":"brentford",
    "crystal palace":"crystal_palace","palace":"crystal_palace",
    "everton":"everton",
    "fulham":"fulham",
    "ipswich":"ipswich","ipswich town":"ipswich",
    "leicester":"leicester","leicester city":"leicester",
    "newcastle":"newcastle","newcastle united":"newcastle",
    "nottingham forest":"nottingham_forest","nott'm forest":"nottingham_forest","forest":"nottingham_forest",
    "southampton":"southampton",
    "west ham":"west_ham","west ham united":"west_ham",
    "wolves":"wolves","wolverhampton":"wolves","wolverhampton wanderers":"wolves",
}
def _clean_name(s: str) -> str:
    return (s or "").lower().replace("’","'").replace(".", "").strip()
def to_slug(name: str) -> str:
    key = _clean_name(name)
    return SLUG_ALIASES.get(key, key)

# =========================
# Bivariate Poisson helpers
# =========================
def bivariate_pmf(h, a, lam1, lam2, lam3):
    base = math.exp(-(lam1 + lam2 + lam3))
    tot = 0.0
    for i in range(min(h, a) + 1):
        num = (lam1 ** (h - i)) * (lam2 ** (a - i)) * (lam3 ** i)
        den = math.factorial(h - i) * math.factorial(a - i) * math.factorial(i)
        tot += num / den
    return base * tot

def min_odds_for_pos_ev_on_exchange(p, commission=DEFAULT_EXCH_COMM):
    if p <= 0: return float('inf')
    return 1.0 + (1.0/p - 1.0) / max(1e-6, (1.0 - commission))

def local_map_score(lam1, lam2, rho=0.0, max_goals=6):
    """Pick the highest-probability score in a 3x3 grid around (round(mu_h), round(mu_a))."""
    lam3 = rho * min(lam1, lam2)
    mu_h, mu_a = lam1 + lam3, lam2 + lam3
    Hc, Ac = int(round(mu_h)), int(round(mu_a))
    best = (-1.0, 0, 0)
    for H in range(max(0, Hc-1), min(max_goals, Hc+1)+1):
        for A in range(max(0, Ac-1), min(max_goals, Ac+1)+1):
            p = bivariate_pmf(H, A, lam1, lam2, lam3)
            if p > best[0]:
                best = (p, H, A)
    p, H, A = best
    return H, A, p

def result_probs(lam1, lam2, rho=RHO, max_goals=8):
    """Return most likely result letter and its probability."""
    lam3 = rho * min(lam1, lam2)
    pH = pD = pA = 0.0
    for H in range(max_goals + 1):
        for A in range(max_goals + 1):
            p = bivariate_pmf(H, A, lam1, lam2, lam3)
            if H > A:   pH += p
            elif H == A:pD += p
            else:       pA += p
    s = pH + pD + pA
    if s > 0: pH, pD, pA = pH/s, pD/s, pA/s
    if pH >= pD and pH >= pA: return "H", pH
    if pA >= pH and pA >= pD: return "A", pA
    return "D", pD

# =========================
# Understat scraping (robust)
# =========================
REQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer": "https://understat.com/",
    "Cache-Control": "no-cache",
}

def _extract_json_from_jsonparse(html: str, varname: str = "matchesData"):
    pat = rf"{re.escape(varname)}\s*=\s*JSON\.parse\('(?P<blob>(?:\\.|[^'])*)'\)"
    m = re.search(pat, html, re.S)
    if not m:
        m = re.search(r"JSON\.parse\('(?P<blob>(?:\\.|[^'])*)'\)", html, re.S)
    if not m:
        raise RuntimeError("Couldn't locate Understat matchesData")
    js_blob = m.group("blob")
    decoded = js_blob.encode("utf-8").decode("unicode_escape")
    if "\\x" in decoded or "\\u" in decoded:
        decoded = decoded.encode("utf-8").decode("unicode_escape")
    return json.loads(decoded)

def _load_understat_matches(season="2024", retries: int = 3, backoff: float = 1.0):
    url = f"https://understat.com/league/EPL/{season}"
    s = requests.Session()
    s.headers.update(REQ_HEADERS)
    last_err = None
    for attempt in range(retries):
        try:
            html = s.get(url, timeout=20).text
            data = _extract_json_from_jsonparse(html, "matchesData")
            data.sort(key=lambda x: x.get("date", ""))
            return data
        except Exception as e:
            last_err = e
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Couldn't locate Understat matchesData ({last_err})")

def _team_timeseries_from_matches(matches):
    xg_for = defaultdict(list)
    xg_against = defaultdict(list)
    games_played = defaultdict(int)
    for m in matches:
        home = to_slug(m["h"]["title"])
        away = to_slug(m["a"]["title"])
        xGh  = float(m["xG"]["h"])
        xGa  = float(m["xG"]["a"])
        xg_for[home].append(xGh)
        xg_against[home].append(xGa)
        games_played[home] += 1
        xg_for[away].append(xGa)
        xg_against[away].append(xGh)
        games_played[away] += 1
    return xg_for, xg_against, games_played

def _ewma(seq, span=8):
    if not seq:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    val = seq[0]
    for v in seq[1:]:
        val = alpha * v + (1 - alpha) * val
    return val

def build_current_ewma(season="2024", span=8):
    matches = _load_understat_matches(season)
    xgf, xga, gp = _team_timeseries_from_matches(matches)
    ewma_xg  = {t: _ewma(lst, span=span) for t, lst in xgf.items()}
    ewma_xga = {t: _ewma(lst, span=span) for t, lst in xga.items()}
    return ewma_xg, ewma_xga, gp

def build_last_season_priors(prev_season="2023"):
    matches = _load_understat_matches(prev_season)
    xgf, xga, _ = _team_timeseries_from_matches(matches)
    prior_xg  = {t: (sum(lst)/len(lst) if lst else 0.0) for t, lst in xgf.items()}
    prior_xga = {t: (sum(lst)/len(lst) if lst else 0.0) for t, lst in xga.items()}
    league_prior_xg  = sum(prior_xg.values())  / max(1, len(prior_xg))
    league_prior_xga = sum(prior_xga.values()) / max(1, len(prior_xga))
    return prior_xg, prior_xga, league_prior_xg, league_prior_xga

# =========================
# FPL data (strength + fixtures + players) — keyed by slug
# =========================
def _to_float(x): 
    try: return float(x)
    except: return 0.0

def _expected_minutes(p):
    status = p.get("status", "a")
    chance = p.get("chance_of_playing_next_round")
    if status == "a":
        if isinstance(chance, int): 
            return int(90 * chance/100)
        return 90
    if status == "d":
        if isinstance(chance, int):
            return int(90 * chance/100)
        return 30
    return 0  # i/s/n/u

def _per90(val, minutes):
    m = max(1, int(minutes or 0))
    return _to_float(val) * 90.0 / m

def build_player_availability_multipliers(elements, id2slug):
    by_team = defaultdict(list)
    for p in elements:
        p = dict(p)
        p["team_name"] = id2slug.get(p["team"])  # store slug
        if p["team_name"]:
            by_team[p["team_name"]].append(p)
    attack_mult, def_mult = {}, {}
    for tname, plist in by_team.items():
        attackers = [p for p in plist if p["element_type"] in (3, 4)]
        defenders = [p for p in plist if p["element_type"] in (2,)]
        gks       = [p for p in plist if p["element_type"] in (1,)]
        for p in attackers:
            p["_atk_p90"] = _per90(_to_float(p.get("threat", 0.0)) + _to_float(p.get("creativity", 0.0)), p.get("minutes", 0))
            p["_exp_min"] = _expected_minutes(p)
        for p in defenders + gks:
            p["_def_p90"] = _per90(_to_float(p.get("influence", 0.0)), p.get("minutes", 0))
            p["_exp_min"] = _expected_minutes(p)
        atk_sorted = sorted(attackers, key=lambda x: x.get("_atk_p90", 0.0), reverse=True)[:ATK_TOP_N]
        base_atk = sum(p["_atk_p90"] * 90 for p in atk_sorted) or 1.0
        cur_atk  = sum(p["_atk_p90"] * p["_exp_min"] for p in atk_sorted)
        atk_ratio = max(ATK_CLIP[0], min(ATK_CLIP[1], cur_atk / base_atk))
        attack_mult[tname] = atk_ratio
        gk_best = sorted(gks, key=lambda x: x.get("_def_p90", 0.0), reverse=True)[:1]
        def_best = sorted(defenders, key=lambda x: x.get("_def_p90", 0.0), reverse=True)[:DEF_TOP_N]
        core_def = gk_best + def_best
        base_def = sum(p["_def_p90"] * 90 for p in core_def) or 1.0
        cur_def  = sum(p["_def_p90"] * p["_exp_min"] for p in core_def)
        def_ratio = max(DEF_CLIP[0], min(DEF_CLIP[1], (cur_def / base_def) ** DEF_AVAIL_EXP))
        def_mult[tname] = def_ratio
    return attack_mult, def_mult

def fetch_fpl_data(gw):
    bs = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=20).json()
    teams    = bs["teams"]
    elements = bs["elements"]
    fixtures = requests.get(f"https://fantasy.premierleague.com/api/fixtures/?event={gw}", timeout=20).json()

    # team_id -> slug (from FPL full name)
    id2slug = {t["id"]: to_slug(t["name"]) for t in teams}

    # team strengths keyed by slug
    team_str = {}
    for t in teams:
        slug = id2slug[t["id"]]
        team_str[slug] = {
            "strength_attack_home": t["strength_attack_home"],
            "strength_attack_away": t["strength_attack_away"],
            "strength_defence_home": t["strength_defence_home"],
            "strength_defence_away": t["strength_defence_away"],
        }

    avg_ah = sum(t["strength_attack_home"] for t in teams) / len(teams)
    avg_aa = sum(t["strength_attack_away"] for t in teams) / len(teams)

    # fixture difficulties keyed by (home_slug, away_slug)
    diff_map = {}
    for f in fixtures:
        h = id2slug.get(f["team_h"])
        a = id2slug.get(f["team_a"])
        if h and a:
            diff_map[(h, a)] = (f.get("team_h_difficulty",3), f.get("team_a_difficulty",3))

    atk_mult, def_mult = build_player_availability_multipliers(elements, id2slug)
    return team_str, avg_ah, avg_aa, diff_map, atk_mult, def_mult, id2slug

# =========================
# GUI App
# =========================
class ScorePredictApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Exact Score Predictor (priors + EWMA + FPL availability + in-table odds)")
        self.geometry("1000x660")

        # Data containers (all keyed by slug)
        self.curr_xg = {}
        self.curr_xga = {}
        self.gp = {}
        self.prior_xg = {}
        self.prior_xga = {}
        self.league_prior_xg = 1.2
        self.league_prior_xga = 1.2
        self.blend_xg = {}
        self.blend_xga = {}
        self.league_blend_xga = 1.0

        self.team_str = {}
        self.avg_ah = self.avg_aa = 50.0
        self.diff_map = {}
        self.atk_mult = defaultdict(lambda: 1.0)
        self.def_mult = defaultdict(lambda: 1.0)
        self.avg_dh = 50.0
        self.avg_da = 50.0
        self.id2slug = {}
        self.gw_loaded = None

        # rows cache: item_id -> dict
        self.rows = {}
        self._build_ui()
        threading.Thread(target=self._load_data, daemon=True).start()

    def _build_ui(self):
        frm = ttk.Frame(self); frm.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(frm, text="Fixtures (one per line, e.g. 'Arsenal vs Chelsea'). Double-click a 'Market Odds' cell to enter a price.").pack(anchor=tk.W)
        self.text = tk.Text(frm, height=6); self.text.pack(fill=tk.X)

        btnrow = ttk.Frame(frm); btnrow.pack(pady=6)
        ttk.Button(btnrow, text="Predict & Rank", command=self.on_predict).pack(side=tk.LEFT, padx=4)
        ttk.Button(btnrow, text="Check Data",   command=self.on_check_data).pack(side=tk.LEFT, padx=4)

        # Columns include Result and Result %
        cols = ["Fixture", "Predicted Score", "Result", "Result %", "Model Prob %", "Fair Odds", "Fair O (exch 2%)", "Market Odds", "Edge %", "EV/£"]
        self.cols = cols
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=16)
        for c, w in zip(cols, [220, 140, 60, 80, 100, 100, 120, 100, 90, 80]):
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor=tk.CENTER, width=w)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.tree.bind("<Double-1>", self.on_tree_double_click)

        self.status = tk.StringVar(value="⏳ Loading data…")
        ttk.Label(self, textvariable=self.status).pack(anchor=tk.W, padx=10)

    # ── Data loading
    def _load_data(self):
        try:
            bs = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=20).json()
            evs = bs["events"]
            gw  = next((e["id"] for e in evs if e.get("is_current")), evs[0]["id"])
            self.gw_loaded = gw

            (self.team_str, self.avg_ah, self.avg_aa,
             self.diff_map, self.atk_mult, self.def_mult, self.id2slug) = fetch_fpl_data(gw)

            # Defensive league averages for normalization
            dh_vals = [t["strength_defence_home"] for t in self.team_str.values() if "strength_defence_home" in t]
            da_vals = [t["strength_defence_away"] for t in self.team_str.values() if "strength_defence_away" in t]
            self.avg_dh = sum(dh_vals)/len(dh_vals) if dh_vals else 50.0
            self.avg_da = sum(da_vals)/len(da_vals) if da_vals else 50.0

            # Understat trends (slugs applied inside)
            self.curr_xg, self.curr_xga, self.gp = build_current_ewma(season="2024", span=8)
            (self.prior_xg, self.prior_xga,
             self.league_prior_xg, self.league_prior_xga) = build_last_season_priors(prev_season="2023")

            # Blend priors toward current EWMA based on games played
            all_teams = set(self.curr_xg) | set(self.prior_xg) | set(self.team_str.keys())
            for t in all_teams:
                gp_t = self.gp.get(t, 0)
                w = min(gp_t / 8.0, 1.0)
                pxg  = self.prior_xg.get(t,  self.league_prior_xg)
                pxga = self.prior_xga.get(t, self.league_prior_xga)
                cxg  = self.curr_xg.get(t,  pxg)
                cxga = self.curr_xga.get(t, pxga)
                self.blend_xg[t]  = w * cxg  + (1 - w) * pxg
                self.blend_xga[t] = w * cxga + (1 - w) * pxga

            self.league_blend_xga = sum(self.blend_xga.values()) / max(1, len(self.blend_xga))
            self.status.set("✅ Data loaded. Enter fixtures, then double-click 'Market Odds' to price.")
        except Exception as e:
            self.status.set(f"❌ Load error: {e}")

    # ── Prediction
    def on_predict(self):
        if not self.blend_xg:
            messagebox.showwarning("Please wait", "Data still loading…")
            return

        # preserve any already-entered odds by fixture key
        prev_odds = {}
        for iid, r in self.rows.items():
            prev_odds[r["Fixture"]] = r.get("Odds")

        self.tree.delete(*self.tree.get_children())
        self.rows.clear()

        lines = [l.strip() for l in self.text.get("1.0", tk.END).splitlines() if l.strip()]
        tmp_rows = []

        # Gentle, symmetric FD map (neutral at 3)
        FD_FACTOR = {1: 1.10, 2: 1.05, 3: 1.00, 4: 0.95, 5: 0.90}

        for line in lines:
            if "vs" not in line: 
                continue
            home, away = [s.strip() for s in line.split("vs", 1)]
            h, a = to_slug(home), to_slug(away)   # canonical slugs

            # xG/xGA baselines (blended)
            xg_h  = self.blend_xg.get(h, self.league_prior_xg)
            xg_a  = self.blend_xg.get(a, self.league_prior_xg)
            xga_h = self.blend_xga.get(h, self.league_prior_xga)
            xga_a = self.blend_xga.get(a, self.league_prior_xga)

            # Promoted / missing-prior adjustments
            if h not in self.prior_xg:
                xg_h  *= PROM_ATK
                xga_h *= PROM_DEF
            if a not in self.prior_xg:
                xg_a  *= PROM_ATK
                xga_a *= PROM_DEF

            # FPL strengths (attack & DEFENCE normalized)
            th = self.team_str.get(h, {})
            ta = self.team_str.get(a, {})
            sa_h = (th.get("strength_attack_home", self.avg_ah) / self.avg_ah) if self.avg_ah else 1.0
            sa_a = (ta.get("strength_attack_away", self.avg_aa) / self.avg_aa) if self.avg_aa else 1.0
            sd_h = (th.get("strength_defence_home", self.avg_dh) / self.avg_dh) if self.avg_dh else 1.0
            sd_a = (ta.get("strength_defence_away", self.avg_da) / self.avg_da) if self.avg_da else 1.0

            # Understat defence dampening vs league
            df_h = (xga_h / (self.league_blend_xga or 1.0)) ** DEF_EXP
            df_a = (xga_a / (self.league_blend_xga or 1.0)) ** DEF_EXP

            # Fixture difficulty (gentle; neutral if not found)
            diff = self.diff_map.get((h, a))
            if diff:
                diff_h, diff_a = diff
                factor_h = FD_FACTOR.get(int(diff_h), 1.0)
                factor_a = FD_FACTOR.get(int(diff_a), 1.0)
            else:
                factor_h = factor_a = 1.0

            # Team-level goal rates (pre player layer)
            # Opponent defensive strength reduces goals: use 1/sd_*
            lam1 = xg_h * sa_h * (1.0 / max(1e-6, sd_a)) * df_a * factor_h
            lam2 = xg_a * sa_a * (1.0 / max(1e-6, sd_h)) * df_h * factor_a

            # Player availability layer — nudge own attack only
            if USE_PLAYER_LAYER:
                lam1 *= self.atk_mult.get(h, 1.0)
                lam2 *= self.atk_mult.get(a, 1.0)

            # Non-sum-preserving home advantage (modest)
            lam1 *= HOME_ADV

            # Strength ratio tilt (favourite separation), using current rates as proxy
            r = max(1e-6, lam2 / max(1e-6, lam1))
            if r > 1.0:
                tilt = min(r ** K_TILT, TILT_CAP)         # away stronger (cap applied)
                lam2 *= tilt
                lam1 /= tilt
            else:
                tilt = min((1.0 / r) ** K_TILT, TILT_CAP) # home stronger (cap applied)
                lam1 *= tilt
                lam2 /= tilt

            # === LOCAL-MAP predicted score (3x3 around expected goals) ===
            H_pred, A_pred, p_pred = local_map_score(lam1, lam2, rho=RHO, max_goals=MAX_GOALS)
            fair_odds_nv = (1.0 / p_pred) if p_pred > 0 else float('inf')
            fair_odds_ex = min_odds_for_pos_ev_on_exchange(p_pred, commission=DEFAULT_EXCH_COMM)

            # Most-likely match result + its probability
            res_letter, res_prob = result_probs(lam1, lam2, rho=RHO, max_goals=8)

            fixture_key = f"{home.strip()} vs {away.strip()}"  # keep user formatting for display
            tmp_rows.append({
                "Fixture": fixture_key,
                "Pred": f"{H_pred} - {A_pred}",
                "Res": res_letter,
                "ResPct": res_prob,   # store as 0–1
                "Prob": p_pred,
                "Fair": fair_odds_nv,
                "FairEx": fair_odds_ex,
                "Odds": prev_odds.get(fixture_key)
            })

        # default sort by Prob desc (until odds entered)
        tmp_rows.sort(key=lambda r: r["Prob"], reverse=True)

        for r in tmp_rows:
            edge_str, ev_str, odds_str = "", "", ""
            if r.get("Odds"):
                odds = float(r["Odds"])
                imp = 1.0 / odds
                edge_pct = (r["Prob"] - imp) * 100.0
                ev = r["Prob"] * (odds - 1.0) - (1.0 - r["Prob"])
                edge_str = f"{edge_pct:.1f}"
                ev_str = f"{ev:.3f}"
                odds_str = f"{odds:.2f}"

            iid = self.tree.insert(
                "", tk.END,
                values=(
                    r["Fixture"],
                    r["Pred"],
                    r["Res"],
                    f"{r['ResPct']*100:.1f}",
                    f"{r['Prob']*100:.1f}",
                    f"{r['Fair']:.2f}",
                    f"{r['FairEx']:.2f}",
                    odds_str,
                    (edge_str + "%" if edge_str else ""),
                    ev_str
                )
            )
            self.rows[iid] = r

    # ── Editing odds in-table
    def on_tree_double_click(self, event):
        region = self.tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        column = self.tree.identify_column(event.x)  # '#1', '#2', ...
        col_index = int(column.replace("#", "")) - 1
        col_name = self.cols[col_index]
        if col_name != "Market Odds":
            return

        iid = self.tree.identify_row(event.y)
        if not iid:
            return
        r = self.rows.get(iid)
        if not r:
            return

        # Ask for odds
        cur = "" if not r.get("Odds") else str(r["Odds"])
        ans = simpledialog.askstring("Market Odds", f"Enter decimal odds for:\n{r['Fixture']}  [{r['Pred']}]", initialvalue=cur, parent=self)
        if ans is None:
            return
        try:
            odds = float(ans)
            if odds <= 1.0:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid odds", "Please enter decimal odds > 1.0.")
            return

        r["Odds"] = odds
        # recompute edge + EV
        imp = 1.0 / odds
        edge_pct = (r["Prob"] - imp) * 100.0
        ev = r["Prob"] * (odds - 1.0) - (1.0 - r["Prob"])

        # update row display (use column names so indexes stay correct)
        vals = list(self.tree.item(iid, "values"))
        vals[self.cols.index("Market Odds")] = f"{odds:.2f}"
        vals[self.cols.index("Edge %")]      = f"{edge_pct:.1f}%"
        vals[self.cols.index("EV/£")]        = f"{ev:.3f}"
        self.tree.item(iid, values=tuple(vals))

        # resort: rows with odds -> sort by Edge desc; else keep by Prob
        self._resort()

    def _resort(self):
        # pull all rows into list with numeric sort keys
        entries = []
        for iid, r in self.rows.items():
            edge = None
            if r.get("Odds"):
                try:
                    edge = (r["Prob"] - 1.0 / float(r["Odds"])) * 100.0
                except Exception:
                    edge = None
            entries.append((iid, r, edge))

        if any(e is not None for _, _, e in entries):
            # sort by Edge desc; rows without odds sink
            entries.sort(key=lambda x: (-1e-9 if x[2] is None else x[2]), reverse=True)
        else:
            entries.sort(key=lambda x: x[1]["Prob"], reverse=True)

        # reinsert in new order
        for iid, _, _ in entries:
            self.tree.move(iid, "", "end")

    # ── Data check popup (pings FPL + Understat)
    def on_check_data(self):
        def _run():
            lines = []
            cur = None
            events = []
            # FPL bootstrap
            try:
                r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=20)
                r.raise_for_status()
                bs = r.json()
                events = bs.get("events", [])
                cur = next((e for e in events if e.get("is_current")), None)
                nxt  = next((e for e in events if e.get("is_next")), None)
                teams_cnt = len(bs.get("teams", []))
                elems_cnt = len(bs.get("elements", []))
                lines.append(f"FPL bootstrap: OK (teams {teams_cnt}, players {elems_cnt})")
                if cur:
                    lines.append(f"  Current GW: {cur['id']}  deadline: {cur.get('deadline_time','?')}")
                if nxt:
                    lines.append(f"  Next    GW: {nxt['id']}  deadline: {nxt.get('deadline_time','?')}")
            except Exception as e:
                lines.append(f"FPL bootstrap: ERROR → {e}")

            # FPL fixtures for loaded GW (or current if missing)
            try:
                gw = self.gw_loaded if self.gw_loaded is not None else (cur["id"] if cur else (events[0]["id"] if events else 1))
                rf = requests.get(f"https://fantasy.premierleague.com/api/fixtures/?event={gw}", timeout=20)
                rf.raise_for_status()
                fixtures = rf.json()
                lines.append(f"FPL fixtures (GW {gw}): OK (fixtures {len(fixtures)})")
            except Exception as e:
                lines.append(f"FPL fixtures: ERROR → {e}")

            # Understat EPL season page
            try:
                url = "https://understat.com/league/EPL/2024"
                html = requests.get(url, headers={"User-Agent":"Mozilla/5.0","Accept":"text/html,application/xhtml+xml"}, timeout=20).text
                data = _extract_json_from_jsonparse(html, "matchesData")
                n = len(data)
                last_date = ""
                try:
                    last_date = max(m.get("date","") for m in data if m.get("date"))
                except:
                    pass
                teams = set()
                for m in data[:2000]:
                    teams.add(to_slug(m["h"]["title"]))
                    teams.add(to_slug(m["a"]["title"]))
                lines.append(f"Understat: OK (matches {n}, teams {len(teams)}, last date {last_date or 'n/a'})")
            except Exception as e:
                lines.append(f"Understat: ERROR → {e}")

            msg = "\n".join(lines)
            self.after(0, lambda: messagebox.showinfo("Data Check", msg, parent=self))
        threading.Thread(target=_run, daemon=True).start()

if __name__ == "__main__":
    ScorePredictApp().mainloop()
