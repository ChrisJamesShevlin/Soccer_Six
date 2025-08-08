import tkinter as tk
from tkinter import ttk, messagebox
import threading, requests, re, json, math, time
from collections import defaultdict

# =========================
# Tunables
# =========================
DEF_EXP        = 0.90   # defence dampening exponent on xGA
FAV_BOOST      = 1.30   # favourite boost
HOME_FAV_BOOST = 1.20   # extra if favourite is home
RHO            = 0.5    # bivariate Poisson correlation-ish
MAX_GOALS      = 6

# Player availability layer (FPL)
USE_PLAYER_LAYER   = True
ATK_TOP_N          = 5        # best MIDs/FWDs considered "core attackers"
DEF_TOP_N          = 4        # best DEFs (plus GK) considered "core defenders"
ATK_CLIP           = (0.85, 1.15)
DEF_CLIP           = (0.90, 1.15)
DEF_AVAIL_EXP      = 1.00     # >1 makes missing defenders hurt more

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

def pick_scoreline(lam1, lam2, rho=RHO, max_goals=MAX_GOALS):
    lam3 = rho * min(lam1, lam2)
    best, best_p = (0, 0), 0.0
    for H in range(max_goals + 1):
        for A in range(max_goals + 1):
            p = bivariate_pmf(H, A, lam1, lam2, lam3)
            if p > best_p:
                best, best_p = (H, A), p
    return best

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
    """
    Extract JSON payload from: varname = JSON.parse('<js string>');
    Handles escaped quotes and \\x/\\u sequences via double unicode_escape decode.
    """
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
        home = m["h"]["title"]
        away = m["a"]["title"]
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
# FPL data (strength + fixtures + players)
# =========================
def _to_float(x): 
    try: return float(x)
    except: return 0.0

def _expected_minutes(p):
    """Rough expected minutes from FPL status + chance_of_playing_next_round."""
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
    return 0  # i/s/n/u -> not expected

def _per90(val, minutes):
    m = max(1, int(minutes or 0))
    return _to_float(val) * 90.0 / m

def build_player_availability_multipliers(elements, teams):
    """
    Returns:
      attack_mult[name_lower] in [ATK_CLIP]
      def_mult[name_lower]    in [DEF_CLIP]  (bigger = stronger defence available)
    """
    id2name = {t["id"]: t["name"].lower() for t in teams}
    by_team = defaultdict(list)
    for p in elements:
        p = dict(p)  # shallow copy
        p["team_name"] = id2name.get(p["team"], None)
        if p["team_name"]:
            by_team[p["team_name"]].append(p)

    attack_mult = {}
    def_mult = {}

    for tname, plist in by_team.items():
        # split by role
        attackers = [p for p in plist if p["element_type"] in (3, 4)]  # MID/FWD
        defenders = [p for p in plist if p["element_type"] in (2,)]    # DEF
        gks       = [p for p in plist if p["element_type"] in (1,)]    # GK

        # attacker metric: (threat + creativity) per90
        for p in attackers:
            p["_atk_p90"] = _per90(_to_float(p.get("threat", 0.0)) + _to_float(p.get("creativity", 0.0)), p.get("minutes", 0))
            p["_exp_min"] = _expected_minutes(p)

        # defender metric: influence per90
        for p in defenders + gks:
            p["_def_p90"] = _per90(_to_float(p.get("influence", 0.0)), p.get("minutes", 0))
            p["_exp_min"] = _expected_minutes(p)

        # ---- Attack availability
        atk_sorted = sorted(attackers, key=lambda x: x.get("_atk_p90", 0.0), reverse=True)[:ATK_TOP_N]
        base_atk = sum(p["_atk_p90"] * 90 for p in atk_sorted) or 1.0
        cur_atk  = sum(p["_atk_p90"] * p["_exp_min"] for p in atk_sorted)
        atk_ratio = cur_atk / base_atk
        atk_ratio = max(ATK_CLIP[0], min(ATK_CLIP[1], atk_ratio))
        attack_mult[tname] = atk_ratio

        # ---- Defensive availability (GK + best DEF_TOP_N defenders)
        gk_best = sorted(gks, key=lambda x: x.get("_def_p90", 0.0), reverse=True)[:1]
        def_best = sorted(defenders, key=lambda x: x.get("_def_p90", 0.0), reverse=True)[:DEF_TOP_N]
        core_def = gk_best + def_best
        base_def = sum(p["_def_p90"] * 90 for p in core_def) or 1.0
        cur_def  = sum(p["_def_p90"] * p["_exp_min"] for p in core_def)
        def_ratio = (cur_def / base_def) ** DEF_AVAIL_EXP
        def_ratio = max(DEF_CLIP[0], min(DEF_CLIP[1], def_ratio))
        def_mult[tname] = def_ratio

    return attack_mult, def_mult

def fetch_fpl_data(gw):
    bs = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=20).json()
    teams    = bs["teams"]
    elements = bs["elements"]
    fixtures = requests.get(f"https://fantasy.premierleague.com/api/fixtures/?event={gw}", timeout=20).json()

    id2name = {t["id"]: t["name"].lower() for t in teams}
    team_str = {t["name"].lower(): t for t in teams}
    avg_ah = sum(t["strength_attack_home"] for t in teams) / len(teams)
    avg_aa = sum(t["strength_attack_away"] for t in teams) / len(teams)

    diff_map = {}
    for f in fixtures:
        h = id2name.get(f["team_h"])
        a = id2name.get(f["team_a"])
        if h and a:
            diff_map[(h, a)] = (f.get("team_h_difficulty",3), f.get("team_a_difficulty",3))

    atk_mult, def_mult = build_player_availability_multipliers(elements, teams)
    return team_str, avg_ah, avg_aa, diff_map, atk_mult, def_mult

# =========================
# GUI App
# =========================
class ScorePredictApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Exact Score Predictor (priors + EWMA blend + FPL availability)")
        self.geometry("720x600")

        # Data containers
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

        # Player layer
        self.atk_mult = defaultdict(lambda: 1.0)  # team_name_lower -> multiplier
        self.def_mult = defaultdict(lambda: 1.0)

        self._build_ui()
        threading.Thread(target=self._load_data, daemon=True).start()

    def _build_ui(self):
        frm = ttk.Frame(self); frm.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(frm, text="Fixtures (one per line, e.g. Arsenal vs Chelsea):").pack(anchor=tk.W)
        self.text = tk.Text(frm, height=6); self.text.pack(fill=tk.X)
        ttk.Button(frm, text="Predict Scores", command=self.on_predict).pack(pady=6)

        cols = ["Fixture", "Predicted Score"]
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=12)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor=tk.CENTER, width=340)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.status = tk.StringVar(value="⏳ Loading data…")
        ttk.Label(self, textvariable=self.status).pack(anchor=tk.W, padx=10)

    def _load_data(self):
        try:
            # FPL: current or next GW for fixture difficulty + player layer
            evs = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=20).json()["events"]
            gw  = next((e["id"] for e in evs if e.get("is_current")), evs[0]["id"])
            self.team_str, self.avg_ah, self.avg_aa, self.diff_map, self.atk_mult, self.def_mult = fetch_fpl_data(gw + 1)

            # Understat: current season EWMA (last 8) + games played
            self.curr_xg, self.curr_xga, self.gp = build_current_ewma(season="2024", span=8)

            # Understat: last season priors (means)
            self.prior_xg, self.prior_xga, self.league_prior_xg, self.league_prior_xga = build_last_season_priors(prev_season="2023")

            # Blend (per team): w = min(gp/8, 1)
            all_teams = set(self.curr_xg) | set(self.prior_xg) | set(self.team_str.keys())
            for t in all_teams:
                gp_t = self.gp.get(t, 0)
                w = min(gp_t / 8.0, 1.0)

                pxg  = self.prior_xg.get(t,  self.league_prior_xg)
                pxga = self.prior_xga.get(t, self.league_prior_xga)
                cxg  = self.curr_xg.get(t,  pxg)   # if no current yet, start at prior
                cxga = self.curr_xga.get(t, pxga)

                self.blend_xg[t]  = w * cxg  + (1 - w) * pxg
                self.blend_xga[t] = w * cxga + (1 - w) * pxga

            # League avg of blended xGA (for defence normalisation)
            self.league_blend_xga = sum(self.blend_xga.values()) / max(1, len(self.blend_xga))

            layer = " + player availability" if USE_PLAYER_LAYER else ""
            self.status.set(f"✅ Data loaded, ready to predict{layer}.")
        except Exception as e:
            self.status.set(f"❌ Load error: {e}")

    def on_predict(self):
        if not self.blend_xg:
            messagebox.showwarning("Please wait", "Data still loading…")
            return

        self.tree.delete(*self.tree.get_children())

        lines = [l.strip() for l in self.text.get("1.0", tk.END).splitlines() if l.strip()]
        for line in lines:
            if "vs" not in line: 
                continue
            home, away = [s.strip() for s in line.split("vs", 1)]
            h, a = home.lower(), away.lower()

            # Blended xG/xGA (current-season EWMA + last-season prior)
            xg_h  = self.blend_xg.get(home, self.league_prior_xg)
            xg_a  = self.blend_xg.get(away, self.league_prior_xg)
            xga_h = self.blend_xga.get(home, self.league_prior_xga)
            xga_a = self.blend_xga.get(away, self.league_prior_xga)

            # FPL team strength (normalised to league avg)
            th = self.team_str.get(h, {})
            ta = self.team_str.get(a, {})
            sa_h = th.get("strength_attack_home", self.avg_ah) / self.avg_ah
            sa_a = ta.get("strength_attack_away", self.avg_aa) / self.avg_aa

            # Defence factors from xGA (lower is better → reduces opponent λ)
            df_h_raw = xga_h / (self.league_blend_xga or 1.0)
            df_a_raw = xga_a / (self.league_blend_xga or 1.0)
            df_h = df_h_raw ** DEF_EXP
            df_a = df_a_raw ** DEF_EXP

            # Fixture difficulty scaling: 1 (easy) → 1.0, 5 (hard) → 0.2
            diff_h, diff_a = self.diff_map.get((h, a), (3, 3))
            factor_h = (6 - diff_h) / 5.0
            factor_a = (6 - diff_a) / 5.0

            # Base poisson rates
            lam1 = xg_h * sa_h * df_a * factor_h
            lam2 = xg_a * sa_a * df_h * factor_a

            # ---- Player availability multipliers (attack & defence)
            if USE_PLAYER_LAYER:
                am_h = self.atk_mult.get(h, 1.0)
                am_a = self.atk_mult.get(a, 1.0)
                dm_h = self.def_mult.get(h, 1.0)  # bigger = stronger defence available
                dm_a = self.def_mult.get(a, 1.0)

                lam1 *= am_h * (1.0 / max(1e-6, dm_a))
                lam2 *= am_a * (1.0 / max(1e-6, dm_h))

            # Favourite & home-favourite boosts at lambda-level
            if lam1 > lam2:
                lam1 *= FAV_BOOST * HOME_FAV_BOOST
                lam2 *= FAV_BOOST
            else:
                lam2 *= FAV_BOOST

            # Pick most likely exact score
            H, A = pick_scoreline(lam1, lam2, rho=RHO, max_goals=MAX_GOALS)
            self.tree.insert("", tk.END, values=(f"{home} vs {away}", f"{H} - {A}"))

if __name__ == "__main__":
    ScorePredictApp().mainloop()
