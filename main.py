import tkinter as tk
from tkinter import ttk, messagebox
import threading
import requests, re, json, math
from collections import defaultdict

# --- Bivariate Poisson helper ---
def bivariate_pmf(h, a, lam1, lam2, lam3):
    base = math.exp(-(lam1 + lam2 + lam3))
    tot = 0.0
    for i in range(min(h, a) + 1):
        num = (lam1 ** (h - i)) * (lam2 ** (a - i)) * (lam3 ** i)
        den = math.factorial(h - i) * math.factorial(a - i) * math.factorial(i)
        tot += num / den
    return base * tot

def predict_scoreline_bivar(xg_h, xg_a,
                            sa_h, df_h_raw, sa_a, df_a_raw,
                            diff_h, diff_a,
                            rho=0.5, max_goals=6):
    # apply stronger favourite & home boosts:
    FAV_BOOST      = 1.3    # boost for the favourite team
    HOME_FAV_BOOST = 1.2    # extra boost if favourite is at home

    # defence exponent more aggressive
    DEF_EXP = 0.90

    # normalise defence
    df_h = df_h_raw ** DEF_EXP
    df_a = df_a_raw ** DEF_EXP

    # fixture difficulty scaling (soften less-good fixtures less)
    # old: (5-d)/4.0   new: (6-d)/5.0
    factor_h = (6 - diff_h) / 5.0
    factor_a = (6 - diff_a) / 5.0

    # base lambdas
    lam1 = xg_h * sa_h * df_a * factor_h
    lam2 = xg_a * sa_a * df_h * factor_a

    # determine which side is "favourite"
    if lam1 > lam2:
        lam1 *= FAV_BOOST * HOME_FAV_BOOST
        lam2 *= FAV_BOOST
    else:
        lam2 *= FAV_BOOST
        # if away is favourite at home? no extra

    # add correlation term
    lam3 = rho * min(lam1, lam2)

    # pick most likely scoreline
    best, best_p = (0, 0), 0.0
    for H in range(max_goals + 1):
        for A in range(max_goals + 1):
            p = bivariate_pmf(H, A, lam1, lam2, lam3)
            if p > best_p:
                best, best_p = (H, A), p
    return best

# --- Fetch Understat xG + compute xGA (rolling 8-match EWMA) ---
def fetch_understat_data(season="2024"):
    url = f"https://understat.com/league/EPL/{season}"
    r = requests.get(url)
    text = r.text
    m = re.search(r"JSON\.parse\('(?P<data>.+?)'\)", text) or \
        re.search(r'JSON\.parse\("(?P<data>.+?)"\)', text)
    if not m:
        raise RuntimeError("couldn't find matchesData in Understat HTML")
    raw = m.group("data")
    js = raw.encode('utf-8').decode('unicode_escape')
    matches = json.loads(js)

    # compute EWMA xG and xGA per team over last 8 matches
    alpha = 2 / (8 + 1)
    team_xg  = defaultdict(lambda: [])
    team_xga = defaultdict(lambda: [])
    order = []  # preserve match order

    for m in matches:
        order.append(m)
    # only last 8 per team, but EWMA naturally de-weights old:
    for m in order:
        h = m["h"]["title"]
        a = m["a"]["title"]
        xg_h = float(m["xG"]["h"])
        xg_a = float(m["xG"]["a"])

        # EWMA update
        prev = team_xg[h][-1] if team_xg[h] else xg_h
        team_xg[h].append(prev * (1-alpha) + xg_h * alpha)

        prev = team_xga[h][-1] if team_xga[h] else xg_a
        team_xga[h].append(prev * (1-alpha) + xg_a * alpha)

        prev = team_xg[a][-1] if team_xg[a] else xg_a
        team_xg[a].append(prev * (1-alpha) + xg_a * alpha)

        prev = team_xga[a][-1] if team_xga[a] else xg_h
        team_xga[a].append(prev * (1-alpha) + xg_h * alpha)

    # take latest EWMA
    ewma_xg  = {t: vals[-1] for t, vals in team_xg.items()}
    ewma_xga = {t: vals[-1] for t, vals in team_xga.items()}

    # league averages
    league_avg_xg  = sum(ewma_xg.values())  / len(ewma_xg)
    league_avg_xga = sum(ewma_xga.values()) / len(ewma_xga)

    # index by fixture key
    match_index = {
        (m["h"]["title"].lower(), m["a"]["title"].lower()): m
        for m in matches
    }
    return match_index, ewma_xg, ewma_xga, league_avg_xg, league_avg_xga

# --- Fetch FPL data (attack strengths + fixtures difficulty) ---
def fetch_fpl_data(gw):
    bs = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")\
                  .json()
    teams    = bs["teams"]
    fixtures = requests.get(
        f"https://fantasy.premierleague.com/api/fixtures/?event={gw}"
    ).json()

    team_str = {t["name"].lower(): t for t in teams}
    avg_ah   = sum(t["strength_attack_home"] for t in teams) / len(teams)
    avg_aa   = sum(t["strength_attack_away"] for t in teams) / len(teams)

    diff_map = {}
    for f in fixtures:
        h = next(t["name"].lower() for t in teams if t["id"]==f["team_h"])
        a = next(t["name"].lower() for t in teams if t["id"]==f["team_a"])
        diff_map[(h, a)] = (f["team_h_difficulty"], f["team_a_difficulty"])

    return team_str, avg_ah, avg_aa, diff_map

# --- GUI ---
class ScorePredictApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Exact Score Predictor (tuned)")
        self.geometry("620x580")
        self.match_index = {}
        self.ewma_xg      = {}
        self.ewma_xga     = {}
        self.avg_xg       = self.avg_xga = 1.0
        self.team_str     = {}
        self.avg_ah       = self.avg_aa = 50.0
        self.diff_map     = {}
        self.build_ui()
        self.start_fetch_thread()

    def build_ui(self):
        frm = ttk.Frame(self); frm.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(frm,
            text="Fixtures (one per line, e.g. Arsenal vs Chelsea):"
        ).pack(anchor=tk.W)
        self.txt = tk.Text(frm, height=6); self.txt.pack(fill=tk.X)
        ttk.Button(frm, text="Predict Scores", command=self.on_predict)\
           .pack(pady=5)

        cols = ["Fixture", "Predicted Score"]
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=10)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor=tk.CENTER, width=300)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def start_fetch_thread(self):
        def _fetch():
            try:
                evs = requests.get(
                    "https://fantasy.premierleague.com/api/bootstrap-static/"
                ).json()["events"]
                gw = next((e["id"] for e in evs if e["is_current"]), 1)

                (self.match_index,
                 self.ewma_xg,
                 self.ewma_xga,
                 self.avg_xg,
                 self.avg_xga) = fetch_understat_data("2024")

                (self.team_str,
                 self.avg_ah, self.avg_aa,
                 self.diff_map) = fetch_fpl_data(gw + 1)
            except Exception as e:
                messagebox.showerror("Error","Data fetch failed:\n"+str(e))
        threading.Thread(target=_fetch, daemon=True).start()

    def on_predict(self):
        if not self.match_index:
            messagebox.showwarning("Wait","Data still loading…")
            return

        self.tree.delete(*self.tree.get_children())
        for line in self.txt.get("1.0", tk.END).strip().splitlines():
            if "vs" not in line: continue
            home, away = [s.strip() for s in line.split("vs",1)]
            key = (home.lower(), away.lower())
            m   = self.match_index.get(key)
            if not m:
                pred = "No data"
            else:
                xg_h = float(m["xG"]["h"])
                xg_a = float(m["xG"]["a"])

                sa_h = self.team_str.get(home.lower(),{})\
                       .get("strength_attack_home",self.avg_ah)/self.avg_ah
                sa_a = self.team_str.get(away.lower(),{})\
                       .get("strength_attack_away",self.avg_aa)/self.avg_aa

                df_h = self.ewma_xga.get(home, self.avg_xga)/self.avg_xga
                df_a = self.ewma_xga.get(away, self.avg_xga)/self.avg_xga

                diff_h, diff_a = self.diff_map.get(key, (3,3))

                H, A = predict_scoreline_bivar(
                    xg_h, xg_a,
                    sa_h, df_h, sa_a, df_a,
                    diff_h, diff_a,
                    rho=0.5, max_goals=6
                )
                pred = f"{H} - {A}"

            self.tree.insert("", tk.END, values=(f"{home} vs {away}", pred))

if __name__ == "__main__":
    ScorePredictApp().mainloop()
