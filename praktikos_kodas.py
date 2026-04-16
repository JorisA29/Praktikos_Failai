import warnings
warnings.filterwarnings("ignore", message=r".*Timestamp\.utcnow is deprecated.*")
import yfinance as yf, pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.optimize import minimize

# Duomenų gavimas
akcijos = ['AAPL','MSFT','JPM','JNJ','XOM','WMT','PG','CAT','V','KO']
kainos = yf.download(akcijos, start="2020-01-01", end="2026-01-01", auto_adjust=True, progress=False)["Close"].dropna()
kainos_mokymui = kainos.loc["2020-01-01":"2023-12-31"]
kainos_testavimui = kainos.loc["2024-01-01":"2025-12-31"]
grąžos_mokymui = kainos_mokymui.pct_change().dropna()
grąžos_testavimui = kainos_testavimui.pct_change().dropna()

# Normalizuotų kainų grafikas
norm_kainos = kainos / kainos.iloc[0] * 100
plt.figure(figsize=(14,7))
for stulpelis in norm_kainos: plt.plot(norm_kainos.index, norm_kainos[stulpelis], label=stulpelis)
plt.title("JAV akcijų normalizuotos kainos (2020–2025)")
plt.xlabel("Data"); plt.ylabel("Indeksuota reikšmė (pradžia = 100)")
plt.legend(loc="upper left", bbox_to_anchor=(1,1)); plt.tight_layout(); plt.show()

# Metiniai rodikliai
vid_met_grąža = grąžos_mokymui.mean() * 252
kov_matrica = grąžos_mokymui.cov() * 252
nerizikinga_norma = 0.0
n = len(akcijos)

def portfelio_rodikliai(w, mu, cov):
    grąža = np.dot(w, mu)
    rizika = np.sqrt(w @ cov @ w)
    return grąža, rizika

def neigiamas_sharpe(w, mu, cov, rf):
    grąža, rizika = portfelio_rodikliai(w, mu, cov)
    return 1e6 if rizika == 0 else -(grąža - rf) / rizika

# Maksimalaus Sharpe portfelis
opt = minimize(neigiamas_sharpe, np.ones(n)/n, args=(vid_met_grąža, kov_matrica, nerizikinga_norma),
               method='SLSQP', bounds=[(0,1)]*n, constraints={'type':'eq','fun':lambda w: w.sum()-1})
w_ms = opt.x.copy()
w_ms[abs(w_ms) < 1e-10] = 0
w_ms /= w_ms.sum()
grąža_ms, rizika_ms = portfelio_rodikliai(w_ms, vid_met_grąža, kov_matrica)
sr_ms = (grąža_ms - nerizikinga_norma) / rizika_ms

# Vienodų svorių portfelis
w_ew = np.ones(n) / n
grąža_ew, rizika_ew = portfelio_rodikliai(w_ew, vid_met_grąža, kov_matrica)
sr_ew = (grąža_ew - nerizikinga_norma) / rizika_ew

# Rizikos pariteto portfelis
def rp_tikslo_funkcija(w, cov):
    portf_rizika = np.sqrt(w @ cov @ w)
    mrc = cov @ w / portf_rizika
    rc = w * mrc
    return np.sum((rc - rc.mean())**2)
opt_rp = minimize(rp_tikslo_funkcija, np.ones(n)/n, args=(kov_matrica,), method='SLSQP',
                 bounds=[(0,1)]*n, constraints={'type':'eq','fun':lambda w: w.sum()-1})
w_rp = opt_rp.x.copy()
w_rp[abs(w_rp) < 1e-10] = 0
w_rp /= w_rp.sum()
grąža_rp, rizika_rp = portfelio_rodikliai(w_rp, vid_met_grąža, kov_matrica)
sr_rp = (grąža_rp - nerizikinga_norma) / rizika_rp
rc = w_rp * (kov_matrica @ w_rp) / np.sqrt(w_rp @ kov_matrica @ w_rp)
rc_pct = rc / rc.sum() * 100

# Minimalios rizikos portfelis
opt_mr = minimize(lambda w, cov: np.sqrt(w @ cov @ w), np.ones(n)/n, args=(kov_matrica,),
                  method='SLSQP', bounds=[(0,1)]*n, constraints={'type':'eq','fun':lambda w: w.sum()-1})
w_mr = opt_mr.x.copy()
w_mr[abs(w_mr) < 1e-10] = 0
w_mr /= w_mr.sum()
grąža_mr, rizika_mr = portfelio_rodikliai(w_mr, vid_met_grąža, kov_matrica)
sr_mr = (grąža_mr - nerizikinga_norma) / rizika_mr

# funkcija svorių lentelėms spausdinti
def spausdinti_svorius(pavadinimas, w, papildomas=None):
    df = pd.DataFrame({"Akcija": akcijos, "Svoris": w, "Svoris (%)": w*100}).round({"Svoris":4, "Svoris (%)":2})
    if papildomas is not None: df["Rizikos indėlis (%)"] = papildomas.round(2)
    df = df.sort_values("Svoris", ascending=False).reset_index(drop=True)
    print(f"\n{pavadinimas}:\n", df.to_string(index=False), f"\nSvorių suma: {w.sum():.6f}\n")

spausdinti_svorius("Maksimalaus Sharpe rodiklio portfelio svoriai", w_ms)
print(f"Metinė grąža: {grąža_ms:.4f}, metinė rizika: {rizika_ms:.4f}, Sharpe rodiklis: {sr_ms:.4f}")
spausdinti_svorius("Vienodų svorių portfelio svoriai", w_ew)
print(f"Metinė grąža: {grąža_ew:.4f}, metinė rizika: {rizika_ew:.4f}, Sharpe rodiklis: {sr_ew:.4f}")
spausdinti_svorius("Rizikos pariteto portfelio svoriai", w_rp, rc_pct)
print(f"Metinė grąža: {grąža_rp:.4f}, metinė rizika: {rizika_rp:.4f}, Sharpe rodiklis: {sr_rp:.4f}")
spausdinti_svorius("Minimalios rizikos portfelio svoriai", w_mr)
print(f"Metinė grąža: {grąža_mr:.4f}, metinė rizika: {rizika_mr:.4f}, Sharpe rodiklis: {sr_mr:.4f}")

# Testavimo laikotarpio grąžos
test_grąžos = pd.DataFrame({
    "Vienodi svoriai": grąžos_testavimui @ w_ew,
    "Maksimalus Sharpe": grąžos_testavimui @ w_ms,
    "Rizikos paritetas": grąžos_testavimui @ w_rp,
    "Minimali rizika": grąžos_testavimui @ w_mr
})

def metinė_grąža(r): return r.mean() * 252
def metinė_rizika(r): return r.std() * np.sqrt(252)
def sharpe(r): rizika = metinė_rizika(r); return (metinė_grąža(r) - nerizikinga_norma) / rizika if rizika != 0 else np.nan
def bendra_grąža(r): return (1 + r).prod() - 1
def maks_nuosmukis(r):
    kaup = (1 + r).cumprod()
    return (kaup / kaup.cummax() - 1).min()

rezultatai = pd.DataFrame({
    "Metinė grąža": test_grąžos.apply(metinė_grąža),
    "Metinė rizika": test_grąžos.apply(metinė_rizika),
    "Sharpe rodiklis": test_grąžos.apply(sharpe),
    "Bendra grąža": test_grąžos.apply(bendra_grąža),
    "Maksimalus nuosmukis": test_grąžos.apply(maks_nuosmukis)
}).round(4)

rezultatai_pct = rezultatai.copy()
stulp_proc = ["Metinė grąža", "Metinė rizika", "Bendra grąža", "Maksimalus nuosmukis"]
rezultatai_pct[stulp_proc] = (rezultatai_pct[stulp_proc] * 100).round(2)

# Kaupiamosios grąžos grafikas
kaup_grąža = (1 + test_grąžos).cumprod()
plt.figure(figsize=(12,6))
for stulp in kaup_grąža: plt.plot(kaup_grąža.index, kaup_grąža[stulp], label=stulp)
plt.title("Portfelių bendra grąža testavimo laikotarpiu (2024–2025 m.)")
plt.xlabel("Data"); plt.ylabel("Portfelio vertė"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# Galutinės lentelės
svorių_visi = pd.DataFrame({
    "Akcija": akcijos,
    "Vienodi svoriai (%)": w_ew * 100,
    "Maksimalus Sharpe (%)": w_ms * 100,
    "Rizikos paritetas (%)": w_rp * 100,
    "Minimali rizika (%)": w_mr * 100
}).round(2)

print("\nVeiklos rodiklių lentelė (2024–2025 m.):\n", rezultatai_pct)
print("\nPortfelių svorių lentelė (%):\n", svorių_visi.to_string(index=False))

# Atskirų akcijų statistika
def nuosmukis_iš_kainų(k): return (k / k.cummax() - 1).min()
akcijų_statistika = pd.DataFrame({
    "Metinė vidutinė grąža (%)": vid_met_grąža * 100,
    "Metinė rizika (%)": np.sqrt(np.diag(kov_matrica)) * 100,
    "Maksimalus nuosmukis (%)": kainos_mokymui.apply(nuosmukis_iš_kainų) * 100
}).round(2).sort_values("Metinė vidutinė grąža (%)", ascending=False)

print("\nAtrinktų akcijų statistinės charakteristikos mokymo laikotarpiu:\n", akcijų_statistika)
