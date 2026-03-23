"""
xai_neat_features.py — Analyse XAI : Feature Importance du réseau NEAT Snake
==============================================================================
Équivalent de xai_features.py pour l'algorithme génétique NEAT.

3 analyses :
  1. Permutation Importance  : brouiller chaque feature → chute de score
  2. Poids d'entrée          : influence structurelle de chaque feature
                               (somme |poids| des connexions issues de chaque entrée)
  3. Corrélation features/actions : quelle feature déclenche quelle action ?

Différences vs DQL :
  - États bruts (pixels), non normalisés
  - Pas de couche linéaire fixe → agrégation des poids par nœud d'entrée
  - Le réseau peut avoir des connexions directes entrée→sortie (skip connections)

Usage :
    python xai_neat_features.py                   # tout
    python xai_neat_features.py --permutation     # permutation importance
    python xai_neat_features.py --weights         # analyse des poids
    python xai_neat_features.py --correlation     # corrélation features/actions
    python xai_neat_features.py --model best_model_11
    python xai_neat_features.py --episodes 20
"""

import argparse
import math
import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
from scipy.stats import pearsonr

import neat
import neat.config
import neat.nn
import pygame

import snake as game

# ── Pygame headless ─────────────────────────────
pygame.init()
game.show    = False
game.display = None

# ── Dossier de sortie ───────────────────────────
OUT_DIR = "xai_neat_features"
os.makedirs(OUT_DIR, exist_ok=True)

def out(filename: str) -> str:
    return os.path.join(OUT_DIR, filename)


# ─────────────────────────────────────────────
#  Constantes
# ─────────────────────────────────────────────
FEATURE_NAMES = [
    "Dist. mur  N",  "Dist. mur  NE", "Dist. mur  E",  "Dist. mur  SE",
    "Dist. mur  S",  "Dist. mur  SW", "Dist. mur  W",  "Dist. mur  NW",
    "Dist. food N",  "Dist. food NE", "Dist. food E",  "Dist. food SE",
    "Dist. food S",  "Dist. food SW", "Dist. food W",  "Dist. food NW",
]
ACTION_NAMES  = ["UP ↑", "RIGHT →", "DOWN ↓", "LEFT ←"]
ACTION_COLORS = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"]

BG       = "#0D1117"
PANEL_BG = "#0D1B2A"
GRID_COL = "#1E3A5F"
TEXT_COL = "#CCDDEE"

CMAP_IMPORTANCE = LinearSegmentedColormap.from_list(
    "imp", ["#0D1B2A", "#1A3A5C", "#1F618D", "#2E86C1", "#F39C12", "#E74C3C"]
)
CMAP_CORR = LinearSegmentedColormap.from_list(
    "corr", ["#C0392B", "#922B21", "#1A1A2E", "#1A5276", "#2E86C1"]
)
CMAP_WEIGHT = LinearSegmentedColormap.from_list(
    "wgt", ["#0D1B2A", "#154360", "#1F618D", "#AED6F1", "#EBF5FB"]
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


def apply_style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color="white", fontsize=12,
                 fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=9)
    ax.tick_params(colors="#8899AA", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(axis="x", color=GRID_COL, linewidth=0.5, alpha=0.5)


# ─────────────────────────────────────────────
#  Analyse 1 — Permutation Importance
# ─────────────────────────────────────────────
def run_episode(net, env: NeatSnakeEnv,
                shuffle_feature: int = -1) -> tuple[int, list, list]:
    """Joue un épisode complet. Si shuffle_feature ≥ 0, cette feature est randomisée."""
    state       = env.reset()
    done        = False
    states_log  = []
    actions_log = []

    while not done:
        s = list(state)
        if shuffle_feature >= 0:
            # Remplace par une valeur aléatoire dans la plage observée
            # (0-750 pour murs, 0-1000 pour food alignée)
            s[shuffle_feature] = float(np.random.uniform(0, 800))

        outs   = np.array(net.activate(s))
        action = int(np.argmax(outs))

        states_log.append(s)
        actions_log.append(action)
        state, _, done, info = env.step(action)

    return info["score"], states_log, actions_log


def compute_permutation_importance(net, env: NeatSnakeEnv,
                                   n_episodes: int = 20):
    n_feat = len(FEATURE_NAMES)

    print(f"  [PI] Baseline ({n_episodes} épisodes)…")
    baseline_scores = [run_episode(net, env)[0] for _ in range(n_episodes)]
    baseline_mean   = float(np.mean(baseline_scores))
    print(f"  [PI] Score baseline moyen : {baseline_mean:.2f}")

    drops     = np.zeros(n_feat)
    drops_std = np.zeros(n_feat)

    for fi in range(n_feat):
        shuffled   = [run_episode(net, env, shuffle_feature=fi)[0]
                      for _ in range(n_episodes)]
        mean_sh    = float(np.mean(shuffled))
        drop       = baseline_mean - mean_sh
        drops[fi]     = max(drop, 0.0)
        drops_std[fi] = float(np.std(shuffled))
        print(f"  [PI] Feature {fi:>2} ({FEATURE_NAMES[fi]:<18}) : "
              f"score={mean_sh:.2f}  drop={drop:+.2f}")

    return drops, baseline_mean, drops_std


def plot_permutation_importance(drops: np.ndarray, baseline: float,
                                drops_std: np.ndarray):
    n     = len(FEATURE_NAMES)
    order = np.argsort(drops)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=BG,
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle(
        f"Permutation Importance NEAT — score baseline : {baseline:.2f}\n"
        "Chaque feature est randomisée individuellement ; chute de score = son importance",
        fontsize=14, fontweight="bold", color="white"
    )

    # ── Barplot ────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    norm   = Normalize(vmin=drops.min(), vmax=max(drops.max(), 1e-6))
    colors = [CMAP_IMPORTANCE(norm(drops[order[i]])) for i in range(n)]

    bars = ax.barh(range(n), drops[order], xerr=drops_std[order],
                   color=colors, edgecolor="#1A1A2E",
                   error_kw=dict(ecolor="#AAAAAA", lw=1.2, capsize=3),
                   height=0.72)

    for i, (bar, drop) in enumerate(zip(bars, drops[order])):
        ax.text(drop + 0.02, i, f"{drop:.2f}",
                va="center", ha="left", color=TEXT_COL, fontsize=8)

    sep = sum(1 for i in order if i < 8)
    ax.axhline(y=n - sep - 0.5, color="#F39C12", linewidth=1.2,
               linestyle="--", alpha=0.7)
    ax.text(drops.max() * 0.98, n - sep - 0.3,
            "── murs  /  nourriture ──",
            color="#F39C12", fontsize=8, ha="right", alpha=0.8)

    ax.set_yticks(range(n))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in order],
                       color=TEXT_COL, fontsize=9)
    apply_style(ax, "Chute de score par feature randomisée",
                xlabel="Drop de score moyen (baseline − randomisé)")

    # ── Radar chart top 8 ──────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)
    ax2.set_aspect("equal")

    top8       = order[:8]
    drops_top8 = drops[top8]
    vals       = drops_top8 / (drops_top8.max() + 1e-8)
    labs       = [FEATURE_NAMES[i] for i in top8]
    N          = len(top8)
    angles     = [2 * math.pi * k / N for k in range(N)] + [0]
    vals_r     = list(vals) + [vals[0]]

    for level in [0.25, 0.5, 0.75, 1.0]:
        ring_xs = [level * math.cos(a) for a in angles]
        ring_ys = [level * math.sin(a) for a in angles]
        ax2.plot(ring_xs, ring_ys, color=GRID_COL, linewidth=0.7, alpha=0.6)
        ax2.text(level + 0.04, 0.02,
                 f"{int(level*100)}%\n({level*drops_top8.max():.1f})",
                 color="#7A9CC0", fontsize=6, va="center", alpha=0.9,
                 multialignment="center")

    for a in angles[:-1]:
        ax2.plot([0, math.cos(a)], [0, math.sin(a)],
                 color=GRID_COL, linewidth=0.7, alpha=0.6)

    xs = [v * math.cos(a) for v, a in zip(vals_r, angles)]
    ys = [v * math.sin(a) for v, a in zip(vals_r, angles)]
    ax2.fill(xs, ys, color="#2E86C1", alpha=0.30)
    ax2.plot(xs, ys, color="#4FC3F7", linewidth=2.2)
    ax2.scatter(xs[:-1], ys[:-1], color="#FFD700", s=70, zorder=5)

    for rank, (i, a, lab) in enumerate(zip(range(N), angles[:-1], labs)):
        raw_drop  = drops_top8[i]
        feat_idx  = top8[i]
        lab_color = ACTION_COLORS[0] if feat_idx < 8 else ACTION_COLORS[2]
        ax2.text(1.38 * math.cos(a), 1.38 * math.sin(a),
                 f"#{rank+1} {lab}\ndrop={raw_drop:.2f}",
                 ha="center", va="center",
                 color=lab_color, fontsize=7, fontweight="bold",
                 multialignment="center")

    ax2.set_xlim(-1.75, 1.75)
    ax2.set_ylim(-2.50, 1.75)
    ax2.axis("off")
    ax2.set_title("Top 8 features — Radar d'importance",
                  color="white", fontsize=11, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(out("xai_neat_permutation.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out('xai_neat_permutation.png')}")
    plt.show()


# ─────────────────────────────────────────────
#  Analyse 2 — Poids d'entrée du génome NEAT
# ─────────────────────────────────────────────
def compute_input_weights(genome, config):
    """
    Pour chaque nœud d'entrée (−16 … −1), collecte tous les poids des
    connexions issues de ce nœud (vers hidden et/ou output).
    Retourne :
        weight_sums [16] : somme des |poids| par feature d'entrée
        weight_stds [16] : écart-type des poids par feature d'entrée
        weights_raw [16] : liste de poids bruts par feature
        direct_to_out [16, 4] : poids directs entrée→sortie (0 si absent)
    """
    input_keys  = config.genome_config.input_keys   # [-16, -15, ..., -1]
    output_keys = config.genome_config.output_keys  # [0, 1, 2, 3]
    n_in, n_out = len(input_keys), len(output_keys)

    weight_sums   = np.zeros(n_in)
    weight_stds   = np.zeros(n_in)
    weights_raw   = [[] for _ in range(n_in)]
    direct_to_out = np.zeros((n_in, n_out))

    for (src, dst), conn in genome.connections.items():
        if not conn.enabled:
            continue
        if src in input_keys:
            fi = input_keys.index(src)
            weights_raw[fi].append(conn.weight)
            if dst in output_keys:
                ai = output_keys.index(dst)
                direct_to_out[fi, ai] = conn.weight

    for fi in range(n_in):
        ws = weights_raw[fi]
        if ws:
            weight_sums[fi] = np.sum(np.abs(ws))
            weight_stds[fi] = np.std(ws)

    return weight_sums, weight_stds, weights_raw, direct_to_out


def plot_input_weights(genome, config):
    weight_sums, weight_stds, weights_raw, direct_to_out = \
        compute_input_weights(genome, config)

    n_in  = len(FEATURE_NAMES)
    n_out = 4

    fig = plt.figure(figsize=(20, 10), facecolor=BG)
    fig.suptitle(
        "Analyse des poids d'entrée NEAT — Features utilisées vs ignorées\n"
        "Connexions actives du génome vers les nœuds cachés et de sortie",
        fontsize=14, fontweight="bold", color="white"
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.38, hspace=0.48)

    # ── Haut gauche : somme |poids| par feature ─────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANEL_BG)
    order1 = np.argsort(weight_sums)[::-1]
    norm_c = Normalize(vmin=weight_sums.min(), vmax=weight_sums.max() + 1e-8)
    colors1 = [CMAP_WEIGHT(norm_c(weight_sums[i])) for i in order1]

    ax1.barh(range(n_in), weight_sums[order1], color=colors1,
             edgecolor="#0D1117", height=0.7)
    ax1.set_yticks(range(n_in))
    ax1.set_yticklabels([FEATURE_NAMES[i] for i in order1],
                        color=TEXT_COL, fontsize=8)
    for i, v in enumerate(weight_sums[order1]):
        ax1.text(v + 0.01, i, f"{v:.2f}", va="center",
                 color=TEXT_COL, fontsize=7)
    apply_style(ax1, "Influence structurelle par feature\n(Σ|poids| des connexions sortantes)",
                xlabel="Σ |poids| des connexions de la feature")

    # ── Haut droite : écart-type par feature ────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PANEL_BG)
    order2 = np.argsort(weight_stds)[::-1]
    colors2 = [CMAP_IMPORTANCE(norm_c(weight_stds[i])) for i in order2]

    ax2.barh(range(n_in), weight_stds[order2], color=colors2,
             edgecolor="#0D1117", height=0.7)
    ax2.set_yticks(range(n_in))
    ax2.set_yticklabels([FEATURE_NAMES[i] for i in order2],
                        color=TEXT_COL, fontsize=8)
    for i, v in enumerate(weight_stds[order2]):
        ax2.text(v + 0.001, i, f"{v:.3f}", va="center",
                 color=TEXT_COL, fontsize=7)
    apply_style(ax2, "Dispersion des poids par feature\n(std des poids des connexions sortantes)",
                xlabel="std(poids)")

    # ── Bas gauche : heatmap poids directs entrée→sortie ─
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(PANEL_BG)

    # Masquer les connexions absentes
    mask = direct_to_out == 0
    vabs = np.abs(direct_to_out).max() or 1.0

    im = ax3.imshow(direct_to_out, cmap="RdBu_r",
                    vmin=-vabs, vmax=vabs,
                    aspect="auto", interpolation="nearest")
    for fi in range(n_in):
        for ai in range(n_out):
            v = direct_to_out[fi, ai]
            c = "white" if abs(v) > vabs * 0.4 else "#888888"
            txt = f"{v:+.2f}" if v != 0 else "—"
            ax3.text(ai, fi, txt, ha="center", va="center",
                     color=c, fontsize=7)

    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Poids direct", color=TEXT_COL, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    ax3.set_xticks(range(n_out))
    ax3.set_xticklabels(ACTION_NAMES, color=TEXT_COL, fontsize=9)
    ax3.set_yticks(range(n_in))
    ax3.set_yticklabels(FEATURE_NAMES, color=TEXT_COL, fontsize=7.5)
    ax3.set_title("Connexions directes entrée → sortie\n(poids du génome ; '—' = pas de connexion)",
                  color="white", fontsize=11, fontweight="bold", pad=8)
    ax3.tick_params(colors="#8899AA", labelsize=7)
    for spine in ax3.spines.values():
        spine.set_edgecolor(GRID_COL)

    # ── Bas droite : scatter importance vs dispersion ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PANEL_BG)

    colors_sc = [ACTION_COLORS[0] if i < 8 else ACTION_COLORS[2]
                 for i in range(n_in)]
    ax4.scatter(weight_sums, weight_stds, c=colors_sc, s=90,
                edgecolors="#222244", linewidths=0.8, zorder=3)
    for i, (x, y) in enumerate(zip(weight_sums, weight_stds)):
        ax4.annotate(FEATURE_NAMES[i], (x, y),
                     textcoords="offset points", xytext=(5, 3),
                     color=TEXT_COL, fontsize=6.5, alpha=0.85)

    ax4.axvline(x=np.median(weight_sums), color="#F39C12",
                linestyle="--", linewidth=1, alpha=0.6)
    ax4.axhline(y=np.median(weight_stds), color="#F39C12",
                linestyle="--", linewidth=1, alpha=0.6)
    ax4.text(np.median(weight_sums) * 0.1, weight_stds.max() * 0.93,
             "faible\ninfluence", color="#F39C12", fontsize=7.5, alpha=0.7)
    ax4.text(np.median(weight_sums) * 1.08, weight_stds.max() * 0.93,
             "forte\ninfluence", color="#F39C12", fontsize=7.5, alpha=0.7)

    legend_p = [
        mpatches.Patch(color=ACTION_COLORS[0], label="Distances murs (0–7)"),
        mpatches.Patch(color=ACTION_COLORS[2], label="Distances nourriture (8–15)"),
    ]
    ax4.legend(handles=legend_p, fontsize=8, facecolor="#0D1117",
               edgecolor="#444", labelcolor="white")
    apply_style(ax4, "Importance vs Dispersion des poids",
                xlabel="Σ |poids| (influence totale)",
                ylabel="std(poids) (dispersion)")

    plt.savefig(out("xai_neat_weights.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out('xai_neat_weights.png')}")
    plt.show()


# ─────────────────────────────────────────────
#  Analyse 3 — Corrélation features / actions
# ─────────────────────────────────────────────
def compute_feature_action_correlation(net, env: NeatSnakeEnv,
                                       n_episodes: int = 20):
    all_states  = []
    all_actions = []

    print(f"  [Corr] Collecte sur {n_episodes} épisodes…")
    for ep in range(n_episodes):
        _, states, actions = run_episode(net, env)
        all_states.extend(states)
        all_actions.extend(actions)

    states_arr  = np.array(all_states,  dtype=np.float32)
    actions_arr = np.array(all_actions, dtype=np.int32)
    print(f"  [Corr] {len(actions_arr)} transitions collectées.")

    n_feat = len(FEATURE_NAMES)
    corr_matrix = np.zeros((n_feat, 4))
    for fi in range(n_feat):
        for ai in range(4):
            binary = (actions_arr == ai).astype(float)
            r, _   = pearsonr(states_arr[:, fi], binary)
            corr_matrix[fi, ai] = r if not np.isnan(r) else 0.0

    mean_per_action = np.zeros((4, n_feat))
    std_per_action  = np.zeros((4, n_feat))
    for ai in range(4):
        mask = actions_arr == ai
        if mask.sum() > 0:
            mean_per_action[ai] = states_arr[mask].mean(axis=0)
            std_per_action[ai]  = states_arr[mask].std(axis=0)

    return corr_matrix, mean_per_action, std_per_action


def plot_feature_action_correlation(corr_matrix: np.ndarray,
                                    mean_per_action: np.ndarray,
                                    std_per_action: np.ndarray):
    fig = plt.figure(figsize=(20, 12), facecolor=BG)
    fig.suptitle(
        "Corrélation Features → Actions NEAT  —  Ce qui déclenche chaque décision",
        fontsize=15, fontweight="bold", color="white"
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.40, hspace=0.50)

    # ── Heatmap centrale ───────────────────────
    ax_heat = fig.add_subplot(gs[:, 0])
    ax_heat.set_facecolor(PANEL_BG)

    vabs = np.abs(corr_matrix).max()
    im   = ax_heat.imshow(corr_matrix, cmap=CMAP_CORR,
                          vmin=-vabs, vmax=vabs,
                          aspect="auto", interpolation="nearest")

    for fi in range(len(FEATURE_NAMES)):
        for ai in range(4):
            v = corr_matrix[fi, ai]
            c = "white" if abs(v) > 0.15 else "#888888"
            ax_heat.text(ai, fi, f"{v:+.2f}", ha="center", va="center",
                         color=c, fontsize=8, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Corrélation de Pearson", color=TEXT_COL, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    ax_heat.set_xticks(range(4))
    ax_heat.set_xticklabels(ACTION_NAMES, color=TEXT_COL, fontsize=9)
    ax_heat.set_yticks(range(len(FEATURE_NAMES)))
    ax_heat.set_yticklabels(FEATURE_NAMES, color=TEXT_COL, fontsize=8)
    ax_heat.set_title("Corrélation\nfeature × action",
                      color="white", fontsize=12, fontweight="bold", pad=10)
    ax_heat.axhline(y=7.5, color="#F39C12", linewidth=1.5,
                    linestyle="--", alpha=0.7)
    for spine in ax_heat.spines.values():
        spine.set_edgecolor(GRID_COL)

    # ── 4 barplots ─────────────────────────────
    positions = [(0, 1), (0, 2), (1, 1), (1, 2)]
    for ai, (row, col) in enumerate(positions):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(PANEL_BG)

        vals  = corr_matrix[:, ai]
        order = np.argsort(np.abs(vals))[::-1]
        ypos  = range(len(FEATURE_NAMES))

        bar_colors = [
            ACTION_COLORS[ai] if v >= 0 else "#E74C3C"
            for v in vals[order]
        ]
        ax.barh(list(ypos), vals[order],
                color=bar_colors, edgecolor="#0D1117",
                alpha=0.85, height=0.7)
        ax.axvline(x=0, color="#AAAAAA", linewidth=1.0, alpha=0.5)
        ax.set_yticks(list(ypos))
        ax.set_yticklabels([FEATURE_NAMES[i] for i in order],
                           color=TEXT_COL, fontsize=7.5)
        ax.set_xlim(-vabs * 1.2, vabs * 1.2)
        ax.set_title(f"Action : {ACTION_NAMES[ai]}",
                     color=ACTION_COLORS[ai], fontsize=11,
                     fontweight="bold", pad=8)
        ax.set_xlabel("Corrélation de Pearson", color=TEXT_COL, fontsize=8)
        ax.tick_params(colors="#8899AA", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(axis="x", color=GRID_COL, linewidth=0.5, alpha=0.5)

    plt.savefig(out("xai_neat_correlation.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out('xai_neat_correlation.png')}")
    plt.show()

    # ── Profil sensoriel par action ─────────────
    _plot_mean_per_action(mean_per_action, std_per_action)


def _plot_mean_per_action(mean_per_action: np.ndarray,
                          std_per_action: np.ndarray):
    fig, axes = plt.subplots(1, 4, figsize=(26, 9), facecolor=BG)
    fig.suptitle(
        "Profil sensoriel par action NEAT  —  Valeur moyenne de chaque feature quand l'agent choisit cette action\n"
        "Les distances sont en pixels (brutes, non normalisées)  |  trait orange = séparation murs / nourriture",
        fontsize=11, fontweight="bold", color="white", y=1.02
    )

    ypos = np.arange(len(FEATURE_NAMES))
    label_colors_y = [ACTION_COLORS[0]] * 8 + [ACTION_COLORS[2]] * 8

    for ai, ax in enumerate(axes):
        ax.set_facecolor(PANEL_BG)
        means = mean_per_action[ai]
        stds  = std_per_action[ai]

        for row_i in range(len(FEATURE_NAMES)):
            bg_c = "#0F2233" if row_i % 2 == 0 else PANEL_BG
            ax.axhspan(row_i - 0.5, row_i + 0.5,
                       color=bg_c, alpha=0.5, zorder=0)

        ax.barh(ypos, means, xerr=stds,
                color=ACTION_COLORS[ai], alpha=0.82,
                edgecolor="#0D1117", height=0.65, zorder=2,
                error_kw=dict(ecolor="#AAAAAA", lw=1, capsize=3))

        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(m + s + 2, i, f"{m:.0f}px",
                    va="center", ha="left", color=TEXT_COL,
                    fontsize=6.5, alpha=0.85)

        ax.set_yticks(ypos)
        ax.set_yticklabels(FEATURE_NAMES, fontsize=8)
        for tick, col in zip(ax.get_yticklabels(), label_colors_y):
            tick.set_color(col)

        ax.axhline(y=7.5, color="#F39C12", linewidth=1.5,
                   linestyle="--", alpha=0.75, zorder=3)

        ax.set_title(f"Action : {ACTION_NAMES[ai]}",
                     color=ACTION_COLORS[ai], fontsize=12,
                     fontweight="bold", pad=10)
        ax.set_xlabel("Distance (pixels)", color=TEXT_COL, fontsize=8)
        ax.tick_params(axis="x", colors="#8899AA", labelsize=8)
        ax.tick_params(axis="y", colors="#8899AA", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(axis="x", color=GRID_COL, linewidth=0.5, alpha=0.5, zorder=1)

    plt.tight_layout()
    plt.savefig(out("xai_neat_mean_per_action.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out('xai_neat_mean_per_action.png')}")
    plt.show()


# ─────────────────────────────────────────────
#  Point d'entrée
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="XAI — Feature Importance NEAT Snake"
    )
    parser.add_argument("--permutation", action="store_true")
    parser.add_argument("--weights",     action="store_true")
    parser.add_argument("--correlation", action="store_true")
    parser.add_argument("--model",    type=str, default="best_model_11")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    run_all = not (args.permutation or args.weights or args.correlation)

    net, genome, config = load_neat_model(args.model)
    env = NeatSnakeEnv()
    print(f"[XAI] Modèle : {args.model}.pkl")
    print(f"[XAI] Nœuds : {len(genome.nodes)} | "
          f"Connexions actives : "
          f"{sum(1 for c in genome.connections.values() if c.enabled)}")

    if run_all or args.weights:
        print("\n[XAI] ── Poids d'entrée du génome ──────────────────")
        plot_input_weights(genome, config)

    if run_all or args.permutation:
        print("\n[XAI] ── Permutation Importance ────────────────────")
        drops, baseline, drops_std = compute_permutation_importance(
            net, env, n_episodes=args.episodes
        )
        plot_permutation_importance(drops, baseline, drops_std)

    if run_all or args.correlation:
        print("\n[XAI] ── Corrélation features × actions ────────────")
        corr, means, stds_a = compute_feature_action_correlation(
            net, env, n_episodes=args.episodes
        )
        plot_feature_action_correlation(corr, means, stds_a)

    print("\n[XAI] Analyse terminée.")


if __name__ == "__main__":
    main()

# python xai_neat_features.py                       # tout (20 épisodes)
# python xai_neat_features.py --weights             # rapide, pas d'épisodes
# python xai_neat_features.py --permutation --episodes 50
# python xai_neat_features.py --correlation --episodes 30
# python xai_neat_features.py --model best_model_8
