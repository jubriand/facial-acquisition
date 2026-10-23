"""
Fenetre de resultats d'une acquisition v2, en deux onglets.

Onglet 1 - Evolution : graphiques empiles, meme axe temporel (0 = lancement).
  1. Temperatures : visage (moyenne), front, nez (meme unite -> un seul graphique, une seule echelle)
  2. Frequence respiratoire (/min)
  3. Frequence cardiaque (bpm)
  Les frequences sont tracees en clair quand leur indice de qualite est insuffisant, en trait plein sinon.
  Survol : ligne verticale commune aux trois graphiques + valeurs a cet instant.

Onglet 2 - Flight vs Landing : pour chaque mesure, distribution en phase de vol et d'atterrissage
  (boites a moustaches + moyenne), et tableau moyenne +/- ecart-type, difference, duree. Seules les images
  ou le visage est suivi (et, pour les frequences, de qualite suffisante) sont retenues. Pas de test
  statistique : les images successives ne sont pas independantes (une p-value serait trompeuse).

Fichiers enregistres a cote du CSV : <nom>.png (onglet 1), <nom>_phases.png et <nom>_phases.csv (onglet 2).
"""

import csv
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# palette de reference (dataviz) : 3 premieres teintes categorielles dans l'ordre fixe, encre et surfaces
SERIES = {"t_face": ("Visage", "#2a78d6"), "t_forehead": ("Front", "#eb6834"), "t_nose": ("Nez", "#1baf7a")}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FREQ_COLOR = "#2a78d6"
LANDING = "#eb6834"  # meme orange que la pastille "landing phase"
RR_Q_MIN, HR_Q_MIN = 0.5, 0.30  # memes seuils de fiabilite que l'interface


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    num = lambda k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])
    data = {k: num(k) for k in ("elapsed_s", "face_tracked", "t_face", "t_forehead", "t_nose",
                                "rr_per_min", "rr_quality", "hr_bpm", "hr_quality")}
    data["unit"] = rows[0]["unit"] if rows else "degC"
    data["phase"] = np.array([r.get("phase") or "flight" for r in rows])
    return data


def _landing_spans(t, phase):
    """Intervalles [debut, fin] (s) pendant lesquels la phase est 'landing'."""
    spans, start = [], None
    for ti, ph in zip(t, phase):
        if ph == "landing" and start is None:
            start = ti
        elif ph != "landing" and start is not None:
            spans.append((start, ti))
            start = None
    if start is not None:
        spans.append((start, t[-1]))
    return spans


def _style(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def _min_span(ax, span):
    """Etendue verticale minimale : evite qu'une variation de 1 /min ressemble a un effondrement."""
    lo, hi = ax.get_ylim()
    if hi - lo < span:
        mid = (lo + hi) / 2
        ax.set_ylim(max(0, mid - span / 2), max(0, mid - span / 2) + span)


def _quality_line(ax, t, y, q, q_min, color):
    """Trait clair partout, trait plein la ou la qualite suffit (identite non portee par la seule couleur)."""
    ax.plot(t, y, color=color, linewidth=2, alpha=0.25)
    good = np.where(q >= q_min, y, np.nan)
    ax.plot(t, good, color=color, linewidth=2)


def build_figure(data: dict, title: str):
    t = data["elapsed_s"]
    unit = "°C" if data["unit"] == "degC" else "valeur brute"
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7.2), height_ratios=[1.4, 1, 1])
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")

    ax = axes[0]
    ends = []
    for key, (label, color) in SERIES.items():
        y = data[key]
        ax.plot(t, y, color=color, linewidth=2, label=label)
        last = np.where(~np.isnan(y))[0]
        if last.size:
            ends.append([y[last[-1]], label, t[last[-1]]])
    _style(ax, f"Temperature ({unit})")
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=8, labelcolor=INK_2)
    # etiquettes directes en fin de courbe, ecartees verticalement pour ne pas se chevaucher
    if ends:
        lo, hi = ax.get_ylim()
        gap = 0.07 * (hi - lo)
        ends.sort(key=lambda e: e[0])  # du bas vers le haut
        ys = []
        for val, _, _ in ends:
            ys.append(val if not ys else max(val, ys[-1] + gap))
        dx = 0.01 * (float(np.nanmax(t)) if t.size else 1.0)
        for (val, label, tx), ly in zip(ends, ys):
            ax.text(tx + dx, ly, f"{label} {val:.1f}", va="center", fontsize=8, color=INK_2, clip_on=False)

    _quality_line(axes[1], t, data["rr_per_min"], data["rr_quality"], RR_Q_MIN, FREQ_COLOR)
    _style(axes[1], "Freq. respiratoire (/min)")
    _min_span(axes[1], 10)
    _quality_line(axes[2], t, data["hr_bpm"], data["hr_quality"], HR_Q_MIN, FREQ_COLOR)
    _style(axes[2], "Freq. cardiaque (bpm)")
    _min_span(axes[2], 40)
    axes[2].set_xlabel("Temps depuis le lancement de l'acquisition (s)", color=INK_2, fontsize=9)
    axes[2].set_xlim(0, max(float(np.nanmax(t)) if t.size else 1.0, 1.0))
    # phases de vol : bandes orange claires pendant la phase "landing", libellees sur le 1er graphique
    for a, b in _landing_spans(t, data["phase"]) if t.size else []:
        for ax in axes:
            ax.axvspan(a, b, color=LANDING, alpha=0.12, lw=0, zorder=0)
        axes[0].text((a + b) / 2, 1.0, "Landing", transform=axes[0].get_xaxis_transform(), ha="center",
                     va="bottom", fontsize=8, color=INK_2)
    note = "trait clair = qualite insuffisante"
    if (data["phase"] == "landing").any():
        note = "bande orange = landing phase   |   " + note
    fig.text(0.99, 0.005, note, ha="right", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.01, 0.97, 0.97))
    return fig, axes


def _hover(fig, axes, data):
    """Ligne verticale commune + valeurs a l'instant survole."""
    t = data["elapsed_s"]
    lines = [ax.axvline(0, color=MUTED, linewidth=0.8, visible=False) for ax in axes]
    box = axes[0].text(0.99, 0.97, "", transform=axes[0].transAxes, ha="right", va="top", fontsize=8,
                       color=INK, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))
    fmt = lambda v, d=1: "--" if np.isnan(v) else f"{v:.{d}f}"

    def on_move(event):
        if event.inaxes not in axes or not t.size:
            return
        i = int(np.clip(np.searchsorted(t, event.xdata), 0, len(t) - 1))
        for ln in lines:
            ln.set_xdata([t[i], t[i]])
            ln.set_visible(True)
        box.set_text(f"t = {t[i]:.1f} s  ({data['phase'][i]} phase)\nVisage {fmt(data['t_face'][i])}   Front {fmt(data['t_forehead'][i])}   "
                     f"Nez {fmt(data['t_nose'][i])}\nFR {fmt(data['rr_per_min'][i], 0)} /min   "
                     f"FC {fmt(data['hr_bpm'][i], 0)} bpm")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)


# --- onglet 2 : comparaison flight / landing -------------------------------------------------------
PHASE_COLORS = {"flight": "#2a78d6", "landing": LANDING}  # memes couleurs que les pastilles de l'interface
PHASE_LABELS = {"flight": "Flight", "landing": "Landing"}


def phase_stats(data: dict) -> list[dict]:
    """Valeurs retenues par mesure et par phase + statistiques descriptives."""
    unit = "°C" if data["unit"] == "degC" else "brut"
    metrics = [("t_face", "Temperature visage", unit, None, None),
               ("t_forehead", "Temperature front", unit, None, None),
               ("t_nose", "Temperature nez", unit, None, None),
               ("rr_per_min", "Freq. respiratoire", "/min", "rr_quality", RR_Q_MIN),
               ("hr_bpm", "Freq. cardiaque", "bpm", "hr_quality", HR_Q_MIN)]
    t = data["elapsed_s"]
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    tracked = data["face_tracked"] == 1
    out = []
    for key, label, u, qkey, qmin in metrics:
        y = data[key]
        ok = tracked & ~np.isnan(y)
        if qkey:
            ok &= data[qkey] >= qmin
        row = {"key": key, "label": label, "unit": u}
        for ph in ("flight", "landing"):
            vals = y[ok & (data["phase"] == ph)]
            row[ph] = vals
            row[ph + "_mean"] = float(vals.mean()) if vals.size else np.nan
            row[ph + "_sd"] = float(vals.std(ddof=1)) if vals.size > 1 else np.nan
            row[ph + "_median"] = float(np.median(vals)) if vals.size else np.nan
            row[ph + "_s"] = vals.size * dt  # duree couverte par les valeurs retenues
        row["diff"] = row["landing_mean"] - row["flight_mean"]
        out.append(row)
    return out


def build_phase_figure(stats: list[dict], title: str):
    fig, axes = plt.subplots(1, len(stats), figsize=(11, 3.8))
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    for ax, row in zip(axes, stats):
        groups = [(ph, row[ph]) for ph in ("flight", "landing")]
        present = [(i + 1, ph, v) for i, (ph, v) in enumerate(groups) if v.size]
        if present:
            bp = ax.boxplot([v for _, _, v in present], positions=[p for p, _, _ in present], widths=0.55,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color=INK, linewidth=1.5),
                            whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
            for patch, (_, ph, _) in zip(bp["boxes"], present):
                patch.set_facecolor(PHASE_COLORS[ph])
                patch.set_alpha(0.35)
                patch.set_edgecolor(PHASE_COLORS[ph])
            for p, ph, v in present:  # moyenne : point blanc cercle de la couleur de la phase
                ax.plot(p, v.mean(), "o", ms=7, mfc="white", mec=PHASE_COLORS[ph], mew=2, zorder=3)
        for i, (ph, v) in enumerate(groups):
            if not v.size:
                ax.text(i + 1, 0.5, "pas de\ndonnee", transform=ax.get_xaxis_transform(), ha="center",
                        va="center", fontsize=8, color=MUTED)
        ax.set_xticks([1, 2], [PHASE_LABELS["flight"], PHASE_LABELS["landing"]])
        ax.set_xlim(0.4, 2.6)
        _style(ax, row["unit"])
        d = 2 if row["unit"] == "°C" else 1
        diff = "" if np.isnan(row["diff"]) else f"\nLanding - Flight : {row['diff']:+.{d}f} {row['unit']}"
        ax.set_title(row["label"] + diff, color=INK_2, fontsize=9, pad=8)
    fig.text(0.99, 0.01, "boite = quartiles, trait = mediane, point = moyenne   |   frequences : valeurs de qualite "
                         "suffisante uniquement", ha="right", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return fig


def save_phase_table(stats: list[dict], path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mesure", "unite", "flight_moyenne", "flight_ecart_type", "flight_mediane", "flight_duree_s",
                    "landing_moyenne", "landing_ecart_type", "landing_mediane", "landing_duree_s",
                    "difference_landing_moins_flight"])
        r = lambda x: "" if np.isnan(x) else f"{x:.3f}"
        for s in stats:
            w.writerow([s["label"], s["unit"], r(s["flight_mean"]), r(s["flight_sd"]), r(s["flight_median"]),
                        f"{s['flight_s']:.1f}", r(s["landing_mean"]), r(s["landing_sd"]), r(s["landing_median"]),
                        f"{s['landing_s']:.1f}", r(s["diff"])])


def _phase_table(parent, stats):
    cols = ("mesure", "flight", "landing", "diff", "duree")
    tv = ttk.Treeview(parent, columns=cols, show="headings", height=len(stats))
    for c, title, w in zip(cols, ("Mesure", "Flight (moy. ± ET)", "Landing (moy. ± ET)", "Landing - Flight",
                                  "Duree flight / landing"), (180, 170, 170, 130, 170)):
        tv.heading(c, text=title)
        tv.column(c, width=w, anchor="w" if c == "mesure" else "center")
    for s in stats:
        d = 2 if s["unit"] == "°C" else 1
        ms = lambda m, sd: "--" if np.isnan(m) else f"{m:.{d}f} ± {0 if np.isnan(sd) else sd:.{d}f} {s['unit']}"
        diff = "--" if np.isnan(s["diff"]) else f"{s['diff']:+.{d}f} {s['unit']}"
        tv.insert("", "end", values=(s["label"], ms(s["flight_mean"], s["flight_sd"]),
                                     ms(s["landing_mean"], s["landing_sd"]), diff,
                                     f"{s['flight_s']:.0f} s / {s['landing_s']:.0f} s"))
    return tv


def show_results(parent: tk.Misc, csv_path: Path):
    """Ouvre la fenetre de resultats (2 onglets) et enregistre les PNG/CSV a cote du CSV. Renvoie le PNG."""
    data = load(csv_path)
    if data["elapsed_s"].size == 0:
        return None
    duration = data["elapsed_s"][-1]
    fig, axes = build_figure(data, f"Acquisition {csv_path.stem}  -  {duration:.0f} s")
    png = csv_path.with_suffix(".png")
    fig.savefig(png, dpi=130, facecolor=SURFACE)

    win = tk.Toplevel(parent)
    win.title(f"Resultats - {csv_path.name}")
    nb = ttk.Notebook(win)
    nb.pack(fill="both", expand=True)

    tab1 = ttk.Frame(nb)
    nb.add(tab1, text="Evolution")
    canvas = FigureCanvasTkAgg(fig, master=tab1)
    NavigationToolbar2Tk(canvas, tab1).update()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    _hover(fig, axes, data)
    canvas.draw()

    tab2 = ttk.Frame(nb, padding=6)
    nb.add(tab2, text="Flight vs Landing")
    figs = [fig]
    if (data["phase"] == "landing").any() and (data["phase"] == "flight").any():
        stats = phase_stats(data)
        fig2 = build_phase_figure(stats, f"Flight vs Landing  -  {csv_path.stem}")
        fig2.savefig(csv_path.with_name(csv_path.stem + "_phases.png"), dpi=130, facecolor=SURFACE)
        save_phase_table(stats, csv_path.with_name(csv_path.stem + "_phases.csv"))
        figs.append(fig2)
        canvas2 = FigureCanvasTkAgg(fig2, master=tab2)
        canvas2.get_tk_widget().pack(fill="both", expand=True)
        canvas2.draw()
        _phase_table(tab2, stats).pack(fill="x", pady=(6, 0))
        ttk.Label(tab2, text="Statistiques descriptives : les images successives ne sont pas independantes, "
                             "aucun test de significativite n'est calcule.", foreground=MUTED).pack(anchor="w", pady=4)
    else:
        ttk.Label(tab2, text="Pas de comparaison possible : l'acquisition ne contient qu'une seule phase "
                             "(utiliser la pastille FLIGHT / LANDING ou la touche P pendant l'acquisition).",
                  wraplength=600, foreground=INK_2).pack(padx=20, pady=40)
    win.protocol("WM_DELETE_WINDOW", lambda: ([plt.close(f) for f in figs], win.destroy()))
    return png
