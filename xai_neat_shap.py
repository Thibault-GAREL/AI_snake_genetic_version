"""
xai_neat_shap.py — Analyse XAI : SHAP pour le réseau NEAT Snake
=================================================================
Équivalent de xai_shap.py pour l'algorithme génétique NEAT.

Différence majeure vs DQL :
  - DQL utilisait shap.DeepExplainer (spécifique PyTorch/TensorFlow)
  - NEAT utilise shap.KernelExplainer (model-agnostic, boîte noire)
    → compatible avec n'importe quelle fonction f(état) → sorties
  → Plus lent, mais fonctionne sur tout modèle

4 visualisations :
  1. Beeswarm plot  — vue globale : impact de chaque feature sur les 4 sorties
  2. Waterfall plot — vue locale : décomposition d'une décision par situation
  3. Force plot     — HTML interactif (une situation à la fois)
  4. Summary heatmap — matrice SHAP [feature × action] + [feature × situation]

Installation requise :
    pip install shap --break-system-packages

Usage :
    python xai_neat_shap.py                    # tout
    python xai_neat_shap.py --beeswarm
    python xai_neat_shap.py --waterfall
    python xai_neat_shap.py --force
    python xai_neat_shap.py --heatmap
    python xai_neat_shap.py --model best_model_11
    python xai_neat_shap.py --episodes 12
    python xai_neat_shap.py --background 100   # taille du background KernelExplainer
    python xai_neat_shap.py --nsamples 200     # nombre d'évaluations SHAP par point
"""

import argparse
import os
import math
import warnings
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches

import neat
import neat.config
import neat.nn
import pygame

warnings.filterwarnings("ignore")

import snake as game

# ── Pygame headless ─────────────────────────────
pygame.init()
game.show    = False
game.display = None

# ── Dossier de sortie ───────────────────────────
OUT_DIR = "xai_neat_shap"
os.makedirs(OUT_DIR, exist_ok=True)

def out(filename: str) -> str:
    return os.path.join(OUT_DIR, filename)


# ─────────────────────────────────────────────
#  Constantes
# ─────────────────────────────────────────────
FEATURE_NAMES = [
    "Mur N",  "Mur NE", "Mur E",  "Mur SE",
    "Mur S",  "Mur SW", "Mur W",  "Mur NW",
    "Food N", "Food NE","Food E", "Food SE",
    "Food S", "Food SW","Food W", "Food NW",
]
ACTION_NAMES  = ["UP ↑", "RIGHT →", "DOWN ↓", "LEFT ←"]
ACTION_COLORS = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"]

SITUATION_NAMES = [
    "Danger N", "Danger E", "Danger S", "Danger W",
    "Food alignée H", "Food alignée V", "Serpent long", "Neutre",
]
SITUATION_COLORS = [
    "#E74C3C", "#F39C12", "#F1C40F", "#2ECC71",
    "#3498DB", "#9B59B6", "#1ABC9C", "#95A5A6",
]

STATE_DIM  = 16
ACTION_DIM = 4

BG       = "#0D1117"
PANEL_BG = "#0D1B2A"
GRID_COL = "#1E3A5F"
TEXT_COL = "#CCDDEE"

CMAP_SHAP = LinearSegmentedColormap.from_list(
    "shap_div", ["#C0392B", "#E8A090", "#F5F5F5", "#90C8E8", "#1A5276"]
)
CMAP_ABS = LinearSegmentedColormap.from_list(
    "shap_abs", ["#0D1B2A", "#154360", "#1F618D", "#F39C12", "#E74C3C"]
)


# ─────────────────────────────────────────────
#  Environnement + chargement
# ─────────────────────────────────────────────
class NeatSnakeEnv:
    def reset(self):
        self.my_snake = game.Manager_snake()
        self.my_snake.add_snake(game.Snake(5 * game.rect_width,
                                           5 * game.rect_height))
        self.food      = game.generated_food(self.my_snake)
        self.score     = 0
        self.iteration = 0
        return self._get_state()

    def _get_state(self):
        s, f = self.my_snake, self.food
        return [
            game.distance_bord_north(s),     game.distance_bord_north_est(s),
            game.distance_bord_est(s),       game.distance_bord_south_est(s),
            game.distance_bord_south(s),     game.distance_bord_south_west(s),
            game.distance_bord_west(s),      game.distance_bord_north_west(s),
            game.distance_food_north(s, f),  game.distance_food_north_est(s, f),
            game.distance_food_est(s, f),    game.distance_food_south_est(s, f),
            game.distance_food_south(s, f),  game.distance_food_south_west(s, f),
            game.distance_food_west(s, f),   game.distance_food_north_west(s, f),
        ]

    def step(self, action):
        if   action == 0 and self.my_snake.direction != "DOWN":
            self.my_snake.direction = "UP"
        elif action == 2 and self.my_snake.direction != "UP":
            self.my_snake.direction = "DOWN"
        elif action == 1 and self.my_snake.direction != "LEFT":
            self.my_snake.direction = "RIGHT"
        elif action == 3 and self.my_snake.direction != "RIGHT":
            self.my_snake.direction = "LEFT"

        head = self.my_snake.list_snake[0]
        if head.x == self.food.x and head.y == self.food.y:
            tail = self.my_snake.list_snake[-1]
            self.my_snake.add_snake(game.Snake(tail.x, tail.y))
            self.food   = game.generated_food(self.my_snake)
            self.score += 1

        alive          = self.my_snake.move()
        self.iteration += 1
        done           = not alive or self.iteration >= game.stop_iteration
        return self._get_state(), float(alive) - 1.0, done, {
            "score": self.score, "iteration": self.iteration
        }


def load_neat_model(model_name: str = "best_model_11"):
    config_path = os.path.join(os.path.dirname(__file__), "config.txt")
    config = neat.config.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path
    )
    try:
        with open(f"{model_name}.pkl", "rb") as f:
            genome = pickle.load(f)
    except FileNotFoundError:
        print(f"[WARN] {model_name}.pkl introuvable — génome aléatoire.")
        genome = list(neat.Population(config).population.values())[0]

    return neat.nn.FeedForwardNetwork.create(genome, config), genome, config


def _classify_situation(state: list, env: NeatSnakeEnv) -> int:
    DANGER_THR = 50
    d_n, d_e, d_s, d_w = state[0], state[2], state[4], state[6]
    food_h = state[10] + state[14]
    food_v = state[8]  + state[12]
    slen   = env.my_snake.lenght
    if 0 < d_n <= DANGER_THR: return 0
    if 0 < d_e <= DANGER_THR: return 1
    if 0 < d_s <= DANGER_THR: return 2
    if 0 < d_w <= DANGER_THR: return 3
    if food_h > 0:             return 4
    if food_v > 0:             return 5
    if slen >= 5:              return 6
    return 7


def apply_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    if title:  ax.set_title(title, color="white", fontsize=11,
                            fontweight="bold", pad=9)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=9)
    ax.tick_params(colors="#8899AA", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)


# ─────────────────────────────────────────────
#  Collecte des états
# ─────────────────────────────────────────────
def collect_states(net, env: NeatSnakeEnv, n_episodes: int = 12):
    all_states, all_actions, all_situations = [], [], []

    for ep in range(n_episodes):
        state = env.reset()
        done  = False
        while not done:
            outs   = np.array(net.activate(state))
            action = int(np.argmax(outs))
            all_states.append(list(state))
            all_actions.append(action)
            all_situations.append(_classify_situation(state, env))
            state, _, done, info = env.step(action)
        print(f"  [Collect] Épisode {ep+1}/{n_episodes} → score {info['score']} "
              f"| total : {len(all_states)}")

    return (
        np.array(all_states,     dtype=np.float32),
        np.array(all_actions,    dtype=np.int32),
        np.array(all_situations, dtype=np.int32),
    )


# ─────────────────────────────────────────────
#  Calcul SHAP avec KernelExplainer
# ─────────────────────────────────────────────
def compute_shap_values(net, states: np.ndarray,
                        background_size: int = 100,
                        nsamples: int = 200) -> tuple:
    """
    Calcule les valeurs SHAP via shap.KernelExplainer (model-agnostic).

    KernelExplainer approche les valeurs de Shapley en échantillonnant
    des combinaisons de features masquées, évaluées sur le background.
    Adaptée aux boîtes noires (pas de gradient requis).

    background_size : nombre d'états de référence (plus grand = plus précis, plus lent)
    nsamples        : évaluations par point SHAP (100–500 selon précision souhaitée)

    Retourne :
        shap_values  : list[np.ndarray [T, 16]] — un array par action
        expected_val : np.ndarray [4] — valeur de base E[f(x)] par action
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "shap non installé.\n"
            "Installez avec : pip install shap --break-system-packages"
        )

    # ── Wrapper du réseau NEAT ─────────────────────
    # KernelExplainer attend f: [N, features] → [N, outputs]
    def neat_predict(states_2d: np.ndarray) -> np.ndarray:
        results = []
        for row in states_2d:
            out = net.activate(row.tolist())
            results.append(out)
        return np.array(results, dtype=np.float32)

    # ── Background : sous-ensemble représentatif ──
    bg_size = min(background_size, len(states))
    bg_idx  = np.random.choice(len(states), bg_size, replace=False)
    bg      = states[bg_idx]
    print(f"  [SHAP] Background : {bg_size} états  |  "
          f"nsamples par point : {nsamples}")
    print(f"  [SHAP] Calcul sur {len(states)} états "
          f"(peut prendre quelques minutes)…")

    # ── KernelExplainer ────────────────────────────
    explainer = shap.KernelExplainer(neat_predict, bg)

    # KernelExplainer.shap_values() retourne une liste [ACTION_DIM] de [T, STATE_DIM]
    shap_values = explainer.shap_values(states, nsamples=nsamples,
                                         silent=True)
    expected    = explainer.expected_value  # [4] ou scalaire

    # ── Normalisation ─────────────────────────────
    T, F = len(states), STATE_DIM
    A    = ACTION_DIM

    if isinstance(expected, (int, float)):
        expected = np.full(A, float(expected))
    else:
        expected = np.array(expected, dtype=np.float32)

    # Cas liste de A arrays [T, F]
    if isinstance(shap_values, (list, tuple)):
        result = []
        for sv in shap_values:
            sv = np.array(sv, dtype=np.float32)
            if sv.shape == (T, F):
                result.append(sv)
            elif sv.ndim == 1 and sv.shape[0] == F:
                result.append(np.tile(sv, (T, 1)))
            else:
                result.append(sv.reshape(T, F))
        shap_values = result
    else:
        # Tableau 3D [T, F, A] ou [A, T, F] …
        arr = np.array(shap_values, dtype=np.float32)
        if arr.ndim == 3:
            if arr.shape == (T, F, A):
                shap_values = [arr[:, :, ai] for ai in range(A)]
            elif arr.shape == (A, T, F):
                shap_values = [arr[ai] for ai in range(A)]
            else:
                shap_values = [arr.reshape(T, F)] * A
        elif arr.ndim == 2:
            shap_values = [arr] * A
        else:
            shap_values = [np.zeros((T, F))] * A

    print(f"  [SHAP] ✓ Shape par action : {shap_values[0].shape}")
    return shap_values, expected


# ─────────────────────────────────────────────
#  Visualisation 1 — Beeswarm plot
# ─────────────────────────────────────────────
def plot_beeswarm(shap_values: list, states: np.ndarray):
    """
    Beeswarm plot pour chaque action (4 subplots).
    Chaque point = un état.  Axe X = valeur SHAP.
    Couleur = valeur de la feature (froid = faible, chaud = élevé).
    """
    fig, axes = plt.subplots(1, 4, figsize=(26, 9), facecolor=BG)
    fig.suptitle(
        "SHAP Beeswarm NEAT — Impact de chaque feature sur chaque sortie\n"
        "Chaque point = un état de jeu  |  "
        "Axe X : valeur SHAP (+ = pousse vers l'action, − = freine)  |  "
        "Couleur : valeur normalisée de la feature",
        fontsize=12, fontweight="bold", color="white", y=1.02
    )

    mean_abs_all = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values],
                           axis=0)
    feat_order   = np.argsort(mean_abs_all)   # croissant → top en haut

    CMAP_FEAT = matplotlib.colormaps.get_cmap("coolwarm")

    for ai, ax in enumerate(axes):
        ax.set_facecolor(PANEL_BG)
        sv = shap_values[ai]
        T  = sv.shape[0]

        for rank, fi in enumerate(feat_order):
            shap_fi = sv[:, fi]
            feat_fi = states[:, fi]

            jitter = np.random.uniform(-0.35, 0.35, size=T)
            y_vals = rank + jitter

            f_min, f_max = feat_fi.min(), feat_fi.max()
            feat_norm = (feat_fi - f_min) / (f_max - f_min + 1e-8)

            ax.scatter(
                shap_fi, y_vals,
                c=feat_norm, cmap=CMAP_FEAT,
                s=6, alpha=0.55, edgecolors="none",
                vmin=0, vmax=1
            )

        ax.axvline(x=0, color="#AAAAAA", linewidth=1.0,
                   linestyle="--", alpha=0.6)
        ax.set_yticks(range(STATE_DIM))
        ax.set_yticklabels([FEATURE_NAMES[fi] for fi in feat_order],
                           color=TEXT_COL, fontsize=8)

        n_mur = sum(1 for fi in feat_order if fi < 8)
        ax.axhline(y=n_mur - 0.5, color="#F39C12",
                   linewidth=1.2, linestyle=":", alpha=0.7)

        apply_style(ax, xlabel="Valeur SHAP (impact sur sortie tanh)")
        ax.set_title(f"Action : {ACTION_NAMES[ai]}",
                     color=ACTION_COLORS[ai], fontsize=12,
                     fontweight="bold", pad=9)
        ax.grid(axis="x", color=GRID_COL, linewidth=0.5, alpha=0.5)

    sm = ScalarMappable(cmap=CMAP_FEAT, norm=Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.008, 0.65])
    cbar    = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Valeur feature (normalisée)",
                   color=TEXT_COL, fontsize=9, rotation=270, labelpad=14)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["Faible", "Moyen", "Élevé"])

    plt.subplots_adjust(right=0.90, wspace=0.45)
    plt.savefig(out("xai_neat_shap_beeswarm.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out('xai_neat_shap_beeswarm.png')}")
    plt.show()


# ─────────────────────────────────────────────
#  Visualisation 2 — Waterfall plot
# ─────────────────────────────────────────────
def plot_waterfall(shap_values: list, states: np.ndarray,
                   situations: np.ndarray, expected: np.ndarray):
    """
    Pour chaque situation, sélectionne l'état représentatif (médian en |SHAP|)
    et trace le waterfall pour l'action dominante.
    """
    from collections import Counter
    n_sit  = len(SITUATION_NAMES)
    n_cols = 4
    n_rows = math.ceil(n_sit / n_cols)

    fig = plt.figure(figsize=(24, 6 * n_rows), facecolor=BG)
    fig.suptitle(
        "SHAP Waterfall NEAT — Décomposition d'une décision par situation\n"
        "Départ = E[f(x)] (valeur de base)  →  Arrivée = sortie tanh prédite",
        fontsize=12, fontweight="bold", color="white", y=1.01
    )

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           wspace=0.45, hspace=0.65)

    for si in range(n_sit):
        row = si // n_cols
        col = si  % n_cols
        ax  = fig.add_subplot(gs[row, col])
        ax.set_facecolor(PANEL_BG)

        mask = situations == si
        if not mask.any():
            ax.set_visible(False)
            continue

        indices = np.where(mask)[0]

        # Action dominante
        action_counts = Counter()
        for idx in indices:
            sv_all = np.array([shap_values[ai][idx] for ai in range(ACTION_DIM)])
            action_counts[int(sv_all.sum(axis=1).argmax())] += 1
        dominant_action = action_counts.most_common(1)[0][0]

        sv_action  = shap_values[dominant_action]
        total_shap = np.abs(sv_action[indices]).sum(axis=1)
        median_val = np.median(total_shap)
        rep_idx    = indices[np.argmin(np.abs(total_shap - median_val))]

        shap_rep = sv_action[rep_idx]
        state_rep = states[rep_idx]
        base_val  = float(expected[dominant_action])

        order     = np.argsort(np.abs(shap_rep))[::-1]
        feat_vals = state_rep[order]
        shap_ord  = shap_rep[order]

        cumulative    = np.zeros(len(order) + 1)
        cumulative[0] = base_val
        for k, s in enumerate(shap_ord):
            cumulative[k + 1] = cumulative[k] + s
        final_val = cumulative[-1]

        bar_bottoms = cumulative[:-1].copy()
        bar_heights = shap_ord.copy()
        for k in range(len(shap_ord)):
            if shap_ord[k] < 0:
                bar_bottoms[k] = cumulative[k + 1]
                bar_heights[k] = -shap_ord[k]

        colors_wf = ["#2E86C1" if s >= 0 else "#C0392B" for s in shap_ord]

        ax.barh(range(len(order)), bar_heights,
                left=bar_bottoms,
                color=colors_wf, edgecolor="#0D1117",
                height=0.68, alpha=0.88)

        for k, (b, h, s) in enumerate(zip(bar_bottoms, bar_heights, shap_ord)):
            x_txt = b + h + (0.01 if s >= 0 else -0.01)
            ha    = "left" if s >= 0 else "right"
            ax.text(x_txt, k, f"{s:+.3f}",
                    va="center", ha=ha, fontsize=6.5,
                    color="#AADDFF" if s >= 0 else "#FFAAAA")

        ax.axvline(x=base_val, color="#F39C12", linewidth=1.2,
                   linestyle="--", alpha=0.8, label=f"E[f(x)]={base_val:.2f}")
        ax.axvline(x=final_val, color="#2ECC71", linewidth=1.4,
                   linestyle="-", alpha=0.8, label=f"f(x)={final_val:.2f}")

        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(
            [f"{FEATURE_NAMES[order[k]]}  [{feat_vals[k]:.0f}px]"
             for k in range(len(order))],
            fontsize=7, color=TEXT_COL
        )

        ax.set_title(
            f"{SITUATION_NAMES[si]}  —  {ACTION_NAMES[dominant_action]}",
            color=SITUATION_COLORS[si], fontsize=10,
            fontweight="bold", pad=7
        )
        ax.set_xlabel("Sortie tanh (contributions cumulées)",
                      color=TEXT_COL, fontsize=8)
        ax.tick_params(colors="#8899AA", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(axis="x", color=GRID_COL, linewidth=0.5, alpha=0.4)

        pos_p = mpatches.Patch(color="#2E86C1", label="Contribution +")
        neg_p = mpatches.Patch(color="#C0392B", label="Contribution −")
        ax.legend(handles=[pos_p, neg_p],
                  fontsize=6.5, facecolor="#0D1117",
                  edgecolor="#444", labelcolor="white",
                  loc="lower right")

    plt.savefig(out("xai_neat_shap_waterfall.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out('xai_neat_shap_waterfall.png')}")
    plt.show()


# ─────────────────────────────────────────────
#  Visualisation 3 — Force plot (HTML)
# ─────────────────────────────────────────────
def plot_force(shap_values: list, states: np.ndarray,
               situations: np.ndarray, expected: np.ndarray):
    """Génère des force plots HTML interactifs par situation."""
    try:
        import shap
    except ImportError:
        print("[SKIP] shap non installé.")
        return

    from collections import Counter
    shap.initjs()

    for si in range(len(SITUATION_NAMES)):
        mask = situations == si
        if not mask.any():
            continue

        indices = np.where(mask)[0][:50]

        action_counts = Counter()
        for idx in indices:
            sv_all = np.array([shap_values[ai][idx] for ai in range(ACTION_DIM)])
            action_counts[int(sv_all.sum(axis=1).argmax())] += 1
        dominant = action_counts.most_common(1)[0][0]

        sv_sit   = shap_values[dominant][indices]
        st_sit   = states[indices]
        base_val = float(expected[dominant])

        html_path = out(
            f"xai_neat_force_sit{si}_"
            f"{SITUATION_NAMES[si].replace(' ','_')}.html"
        )
        try:
            fp = shap.force_plot(
                base_value=base_val,
                shap_values=sv_sit,
                features=st_sit,
                feature_names=FEATURE_NAMES,
                show=False,
                matplotlib=False,
            )
            shap.save_html(html_path, fp)
            print(f"[XAI] Sauvegarde -> {html_path}")
        except Exception as e:
            print(f"[WARN] Force plot situation {si} : {e}")

    # ── Force plot global ─────────────────────────
    from collections import Counter
    all_best = []
    for i in range(len(situations)):
        sv_all = np.array([shap_values[ai][i] for ai in range(ACTION_DIM)])
        all_best.append(int(sv_all.sum(axis=1).argmax()))

    global_dominant = Counter(all_best).most_common(1)[0][0]
    sv_global       = shap_values[global_dominant]
    base_global     = float(expected[global_dominant])

    MAX_HTML = 500
    idx_html = np.linspace(0, len(situations) - 1,
                           min(MAX_HTML, len(situations)), dtype=int)
    html_global = out("xai_neat_force_global.html")
    try:
        fp_global = shap.force_plot(
            base_value=base_global,
            shap_values=sv_global[idx_html],
            features=states[idx_html],
            feature_names=FEATURE_NAMES,
            show=False,
            matplotlib=False,
        )
        shap.save_html(html_global, fp_global)
        print(f"[XAI] Sauvegarde -> {html_global}")
    except Exception as e:
        print(f"[WARN] Force plot global : {e}")


# ─────────────────────────────────────────────
#  Visualisation 4 — Summary heatmap
# ─────────────────────────────────────────────
def plot_summary_heatmap(shap_values: list, states: np.ndarray,
                          situations: np.ndarray):
    """
    4 sous-figures :
    A) |SHAP| moyen [feature × action]
    B) SHAP signé moyen [feature × action]
    C) Barplot importance globale par feature
    D) |SHAP| moyen [feature × situation]
    """
    mean_abs_matrix  = np.zeros((STATE_DIM, ACTION_DIM))
    mean_sign_matrix = np.zeros((STATE_DIM, ACTION_DIM))

    for ai in range(ACTION_DIM):
        mean_abs_matrix[:, ai]  = np.abs(shap_values[ai]).mean(axis=0)
        mean_sign_matrix[:, ai] = shap_values[ai].mean(axis=0)

    global_importance = mean_abs_matrix.mean(axis=1)
    feat_order        = np.argsort(global_importance)   # croissant

    mean_sit_matrix = np.zeros((STATE_DIM, len(SITUATION_NAMES)))
    for si in range(len(SITUATION_NAMES)):
        mask = situations == si
        if not mask.any():
            continue
        for ai in range(ACTION_DIM):
            mean_sit_matrix[:, si] += np.abs(shap_values[ai][mask]).mean(axis=0)
        mean_sit_matrix[:, si] /= ACTION_DIM

    fig = plt.figure(figsize=(24, 14), facecolor=BG)
    fig.suptitle(
        "SHAP Summary NEAT — Vue globale de l'importance des features\n"
        "KernelExplainer (model-agnostic) calculé sur l'ensemble des états collectés",
        fontsize=13, fontweight="bold", color="white"
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.40, hspace=0.55)

    # ── A) |SHAP| moyen feature × action ─────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor(PANEL_BG)
    data_a = mean_abs_matrix[feat_order, :]
    vmax_a = np.percentile(data_a, 97) or 1e-6
    im_a   = ax_a.imshow(data_a, cmap=CMAP_ABS,
                          vmin=0, vmax=vmax_a,
                          aspect="auto", interpolation="nearest")
    for fi_r, fi in enumerate(feat_order):
        for ai in range(ACTION_DIM):
            v = mean_abs_matrix[fi, ai]
            c = "white" if v > vmax_a * 0.5 else TEXT_COL
            ax_a.text(ai, fi_r, f"{v:.3f}",
                      ha="center", va="center", color=c, fontsize=8)
    ax_a.set_xticks(range(ACTION_DIM))
    ax_a.set_xticklabels(ACTION_NAMES, color=TEXT_COL, fontsize=9)
    ax_a.set_yticks(range(STATE_DIM))
    ax_a.set_yticklabels([FEATURE_NAMES[i] for i in feat_order],
                          color=TEXT_COL, fontsize=8)
    ax_a.axhline(y=sum(1 for i in feat_order if i < 8) - 0.5,
                 color="#F39C12", linewidth=1.2, linestyle="--", alpha=0.7)
    cbar_a = plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
    cbar_a.set_label("|SHAP| moyen", color=TEXT_COL, fontsize=8)
    cbar_a.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar_a.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    apply_style(ax_a, title="|SHAP| moyen par feature × action\n"
                             "(plus clair = plus impactant)")

    # ── B) SHAP signé ─────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor(PANEL_BG)
    data_b = mean_sign_matrix[feat_order, :]
    vabs_b = np.abs(data_b).max() or 1e-6
    norm_b = TwoSlopeNorm(vcenter=0, vmin=-vabs_b, vmax=vabs_b)
    im_b   = ax_b.imshow(data_b, cmap=CMAP_SHAP,
                          norm=norm_b, aspect="auto",
                          interpolation="nearest")
    for fi_r, fi in enumerate(feat_order):
        for ai in range(ACTION_DIM):
            v = mean_sign_matrix[fi, ai]
            c = "white" if abs(v) > vabs_b * 0.4 else TEXT_COL
            ax_b.text(ai, fi_r, f"{v:+.3f}",
                      ha="center", va="center", color=c, fontsize=8)
    ax_b.set_xticks(range(ACTION_DIM))
    ax_b.set_xticklabels(ACTION_NAMES, color=TEXT_COL, fontsize=9)
    ax_b.set_yticks(range(STATE_DIM))
    ax_b.set_yticklabels([FEATURE_NAMES[i] for i in feat_order],
                          color=TEXT_COL, fontsize=8)
    ax_b.axhline(y=sum(1 for i in feat_order if i < 8) - 0.5,
                 color="#F39C12", linewidth=1.2, linestyle="--", alpha=0.7)
    cbar_b = plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cbar_b.set_label("SHAP signé moyen", color=TEXT_COL, fontsize=8)
    cbar_b.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar_b.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    apply_style(ax_b, title="SHAP signé moyen par feature × action\n"
                             "(bleu = impact +, rouge = impact −)")

    # ── C) Barplot importance globale ─────────────
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor(PANEL_BG)
    gi_sorted = global_importance[feat_order]
    norm_c    = Normalize(vmin=gi_sorted.min(),
                          vmax=max(gi_sorted.max(), 1e-8))
    colors_c  = [CMAP_ABS(norm_c(v)) for v in gi_sorted]
    ax_c.barh(range(STATE_DIM), gi_sorted, color=colors_c,
              edgecolor="#0D1117", height=0.72)
    for k in range(STATE_DIM):
        ax_c.axhspan(k - 0.5, k + 0.5,
                     color="#0F2233" if k % 2 == 0 else PANEL_BG,
                     alpha=0.4, zorder=0)
    for k, v in enumerate(gi_sorted):
        ax_c.text(v + 0.0003, k, f"{v:.4f}",
                  va="center", color=TEXT_COL, fontsize=7.5)
    ax_c.set_yticks(range(STATE_DIM))
    ax_c.set_yticklabels([FEATURE_NAMES[i] for i in feat_order],
                          color=TEXT_COL, fontsize=8.5)
    ax_c.axhline(y=sum(1 for i in feat_order if i < 8) - 0.5,
                 color="#F39C12", linewidth=1.2, linestyle="--", alpha=0.7)
    n_mur  = sum(1 for i in feat_order if i < 8)
    n_food = STATE_DIM - n_mur
    ax_c.text(gi_sorted.max() * 0.98, n_mur / 2 - 0.5,
              "MURS", color=ACTION_COLORS[0], fontsize=8,
              fontweight="bold", va="center", ha="right", alpha=0.7)
    ax_c.text(gi_sorted.max() * 0.98, n_mur + n_food / 2 - 0.5,
              "FOOD", color=ACTION_COLORS[2], fontsize=8,
              fontweight="bold", va="center", ha="right", alpha=0.7)
    apply_style(ax_c,
                title="Importance SHAP globale (toutes actions)\n"
                      "Rang ↑ = feature la plus influente",
                xlabel="|SHAP| moyen (toutes actions)")
    ax_c.grid(axis="x", color=GRID_COL, linewidth=0.5, alpha=0.5)

    # ── D) Heatmap feature × situation ───────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor(PANEL_BG)
    data_d = mean_sit_matrix[feat_order, :]
    vmax_d = np.percentile(data_d, 97) or 1e-6
    im_d   = ax_d.imshow(data_d, cmap=CMAP_ABS,
                          vmin=0, vmax=vmax_d,
                          aspect="auto", interpolation="nearest")
    ax_d.set_xticks(range(len(SITUATION_NAMES)))
    ax_d.set_xticklabels([s.replace(" ", "\n") for s in SITUATION_NAMES],
                          color=TEXT_COL, fontsize=7.5)
    ax_d.set_yticks(range(STATE_DIM))
    ax_d.set_yticklabels([FEATURE_NAMES[i] for i in feat_order],
                          color=TEXT_COL, fontsize=8)
    ax_d.axhline(y=sum(1 for i in feat_order if i < 8) - 0.5,
                 color="#F39C12", linewidth=1.2, linestyle="--", alpha=0.7)
    for si, col in enumerate(SITUATION_COLORS):
        ax_d.axvline(x=si - 0.5, color=col, linewidth=0.6, alpha=0.4)
    cbar_d = plt.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
    cbar_d.set_label("|SHAP| moyen", color=TEXT_COL, fontsize=8)
    cbar_d.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar_d.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    apply_style(ax_d,
                title="|SHAP| moyen par feature × situation\n"
                      "(quelle feature est cruciale dans quelle situation ?)")

    plt.savefig(out("xai_neat_shap_heatmap.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out('xai_neat_shap_heatmap.png')}")
    plt.show()


# ─────────────────────────────────────────────
#  Point d'entrée
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="XAI — SHAP pour NEAT Snake"
    )
    parser.add_argument("--beeswarm",   action="store_true")
    parser.add_argument("--waterfall",  action="store_true")
    parser.add_argument("--force",      action="store_true")
    parser.add_argument("--heatmap",    action="store_true")
    parser.add_argument("--model",      type=str, default="best_model_11")
    parser.add_argument("--episodes",   type=int, default=12)
    parser.add_argument("--background", type=int, default=100,
                        help="Taille du background KernelExplainer (défaut : 100)")
    parser.add_argument("--nsamples",   type=int, default=200,
                        help="Évaluations SHAP par point (défaut : 200)")
    args = parser.parse_args()

    run_all = not (args.beeswarm or args.waterfall
                   or args.force or args.heatmap)

    try:
        import shap
        print(f"[XAI] shap version : {shap.__version__}")
        print("[XAI] Utilisation de KernelExplainer (model-agnostic)")
    except ImportError:
        print("[ERREUR] shap non installé.")
        print("         pip install shap --break-system-packages")
        return

    net, genome, config = load_neat_model(args.model)
    env = NeatSnakeEnv()
    print(f"[XAI] Modèle : {args.model}.pkl")

    print(f"\n[XAI] Collecte sur {args.episodes} épisode(s)…")
    states, actions, situations = collect_states(
        net, env, n_episodes=args.episodes
    )
    print(f"[XAI] {len(states)} états collectés.\n")

    print("[XAI] Calcul SHAP (KernelExplainer)…")
    shap_values, expected = compute_shap_values(
        net, states,
        background_size=args.background,
        nsamples=args.nsamples
    )

    if run_all or args.beeswarm:
        print("\n[XAI] ── Beeswarm plot ──────────────────────────────")
        plot_beeswarm(shap_values, states)

    if run_all or args.waterfall:
        print("\n[XAI] ── Waterfall plot ─────────────────────────────")
        plot_waterfall(shap_values, states, situations, expected)

    if run_all or args.heatmap:
        print("\n[XAI] ── Summary heatmap ───────────────────────────")
        plot_summary_heatmap(shap_values, states, situations)

    if run_all or args.force:
        print("\n[XAI] ── Force plots HTML ──────────────────────────")
        plot_force(shap_values, states, situations, expected)

    print(f"\n[XAI] Analyse SHAP NEAT terminée. Fichiers dans : {OUT_DIR}/")


if __name__ == "__main__":
    main()

# python xai_neat_shap.py                              # tout (12 épisodes)
# python xai_neat_shap.py --beeswarm                   # le plus informatif
# python xai_neat_shap.py --waterfall
# python xai_neat_shap.py --heatmap
# python xai_neat_shap.py --force
# python xai_neat_shap.py --episodes 20 --background 150 --nsamples 300
# python xai_neat_shap.py --model best_model_8
# AVERTISSEMENT : KernelExplainer est plus lent que DeepExplainer
#   → utiliser --nsamples 50-100 pour un aperçu rapide
