Here is a full **GitHub-ready README.md** for your *Exact Score Predictor + Bet Builder* model — **clean**, **structured**, **technical**, and **without any screenshot section**.

If you want a shorter version, a more visual version, or badges added at the top, just say.

---

# **Exact Score Predictor — EWMA + Priors + FD + Bivariate Poisson + Bet Builder Engine**

A fully-featured football modelling engine that produces:

* Exact score predictions
* 1X2 probabilities
* Fair odds (no-vig + exchange-adjusted)
* A per-match **Bet Builder** evaluator using correlated bivariate-Poisson grid logic
* A full Tkinter GUI with live odds editing
* Automatic scraping of **Understat** (xG time series) and **FPL player fitness layers**

This is a combined **statistical model + betting interface** with a heavy emphasis on explainability, robustness and low-variance probability estimation.

---

## ⭐ Key Features

### 🎯 **1. Exact Score Prediction (Bivariate Poisson)**

The model uses:

* Bivariate Poisson: `lam1`, `lam2`, `lam3 = ρ·min(lam1, lam2)`
* Understat EWMA xG data
* Last-season priors
* Promoted-team adjustments
* Defence exponent scaling
* Player availability layer from FPL
* Fixture difficulty (FPL) weighting
* Home advantage + elite-team peer correction
* Tilt correction for mismatches
* Soft global-mode selection with σ and γ penalty

Final selection is the *global mode* under the softened grid, not merely the raw bivariate highest-cell probability.

---

### 📈 **2. Full Result Probabilities (1X2)**

The model computes:

* Home win %
* Draw %
* Away win %
* Fair odds
* Exchange minimum back odds considering commission
* Market edge and EV per £1 or per stake

---

### 🧠 **3. Combined Offensive & Defensive Model**

Final `lam1` and `lam2` incorporate:

* Blended xG (EWMA + prior)
* Strength attack/defence (FPL)
* Player availability multipliers:

  * Top N attackers
  * Top N defenders + GK
  * Minutes-based expected availability
  * Clipping ranges
* Peer-vs-peer rules (Big 6)

This produces extremely stable per-fixture expected goals.

---

### 🔍 **4. Fully Automated Data Loading**

On startup, the app:

* Loads FPL bootstrap
* Loads FPL fixtures for the current GW
* Scrapes Understat EPL 2024 automatically
* Extracts and decodes `matchesData` using a robust parser
* Builds EWMA curves for every team
* Builds season priors (last season)

All team names from all sources are normalized to a single slug set.

---

### ⚙️ **5. Fixture Difficulty & Peer Clamp System**

* FD from FPL fixtures (values 1–5) blended via weight
* Peer clamp for Big 6 vs Big 6
* Defence scaling clipped to `[0.88, 1.12]`
* Mismatch bump for large gap matches
* Tilt correction (asymmetric boost/suppression depending on relative lam ratio)

---

### 🔢 **6. Bet Builder Engine (Correlated, Grid-Based)**

The model evaluates correlated bet-builder legs:

* BTTS
* Over/Under
* Home/Away outcome
* Combinations (Home+BTTS, Draw+U2.5, Away+O2.5, etc.)

For every builder leg:

* `p = grid_prob(…)` is computed
* Fair odds
* Minimum exchange odds
* Market price entry (via GUI double-click)
* EV per £1 and EV for your flat **£0.25 stake**

Top candidate builders are shown even without odds.

---

### 🧮 **7. GUI Features**

A clean Tkinter GUI provides:

* Text box for listing fixtures (e.g., `Liverpool vs Arsenal`)
* “Predict & Rank” button
* Double-click editable cells for odds (match odds or builder odds)
* Automatic sorting by best edge
* Status bar showing data load status
* Built-in “Check Data” diagnostic (FPL + Understat verification)

Tables included:

#### **Exact Score Tab**

* Fixture
* Predicted score
* Result + result probability
* Model probability
* Fair odds, exchange fair odds
* Market odds
* Edge %
* EV per £1

#### **Bet Builder Tab**

* Suggested builder
* Fair probability
* Fair & exchange minimum odds
* Market builder odds
* Edge
* EV @ £0.25

---

## 📦 Requirements

```
Python 3.9+
requests
tkinter  (usually built-in)
json
math
```

Install missing deps:

```bash
pip install requests
```

Tkinter on Ubuntu:

```bash
sudo apt install python3-tk
```

---

## ▶️ Running the App

```
python3 score_predictor.py
```

The GUI opens immediately and begins scraping Understat / loading FPL data in a background thread.

---

## 🧠 Core Modelling Flow

1. **Understat xG time series → EWMA**
2. **Last-season priors (xG/xGA) → blend factor**
3. **FPL strengths → attack/defence multipliers**
4. **FPL fixture difficulty → soft weight**
5. **Player availability → attack/defence multipliers**
6. **Peer clamp + home advantage**
7. **Tilt correction**
8. **Mismatch bump**
9. **Bivariate Poisson grid → full PMF**
10. **Global-mode softened selection**
11. **Fair odds + EV calculations**
12. **Bet builder probability grid evaluation**

---

## 📚 Files

```
score_predictor.py
README.md
```

---

## 🧩 If You Want More

I can add any of the following:

* CSV export
* Logging of fixture predictions
* Live scraping of bookmaker odds
* Heatmaps for exact score grids
* Integration with your FPL team model
* A dark-mode Tkinter theme (fits your preference)

Just tell me and I’ll extend it.
