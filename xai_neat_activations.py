"""
xai_neat_activations.py — Analyse XAI : Activations internes du réseau NEAT Snake
====================================================================================
Équivalent de xai_activations.py pour l'algorithme génétique NEAT.

Différences clés vs DQL :
  - Réseau NEAT utilise tanh (pas ReLU) → pas de "neurones morts"
    mais des "neurones saturés" (|tanh| > 0.95, toujours à ±1)
  - Topologie variable : les nœuds cachés sont extraits dynamiquement
    depuis le génome (pas de couches fixes 256→128→64)
  - Forward pass custom pour capturer les activations intermédiaires

3 analyses :
  1. Distribution des activations — histogramme, neurones saturés, heatmap temporelle
  2. Neurones spécialisés — quels nœuds cachés s'activent dans quelles situations
  3. t-SNE / UMAP des activations — projection 2D des états de jeu

Usage :
    python xai_neat_activations.py                   # tout
    python xai_neat_activations.py --distribution    # histogramme + saturés
    python xai_neat_activations.py --specialization  # nœuds spécialisés
    python xai_neat_activations.py --tsne            # t-SNE
    python xai_neat_activations.py --umap            # UMAP (plus rapide)
    python xai_neat_activations.py --model best_model_11
    python xai_neat_activations.py --episodes 10
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches

import neat
import neat.config
import neat.nn
import pygame

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import snake as game

# ── Pygame headless ─────────────────────────────
pygame.init()
game.show    = False
game.display = None

# ── Dossier de sortie ───────────────────────────
OUT_DIR = "xai_neat_activations"
os.makedirs(OUT_DIR, exist_ok=True)

def out(filename: str) -> str:
    return os.path.join(OUT_DIR, filename)


# ─────────────────────────────────────────────
#  Constantes
# ─────────────────────────────────────────────
ACTION_NAMES  = ["UP ↑", "RIGHT →", "DOWN ↓", "LEFT ←"]
ACTION_COLORS = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"]

SITUATION_NAMES = [
    "Danger N",  "Danger E",  "Danger S",  "Danger W",
    "Food alignée H", "Food alignée V", "Serpent long (≥5)", "Neutre",
]

BG       = "#0D1117"
PANEL_BG = "#0D1B2A"
GRID_COL = "#1E3A5F"
TEXT_COL = "#CCDDEE"

# tanh saturé → rouge/orange ; actif → vert/bleu
CMAP_SAT = LinearSegmentedColormap.from_list(
    "sat", ["#2ECC71", "#F39C12", "#E74C3C"]
)
CMAP_SPEC = LinearSegmentedColormap.from_list(
    "spec", ["#0D1B2A", "#154360", "#1F618D", "#D4AC0D", "#E74C3C"]
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

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return net, genome, config


# ─────────────────────────────────────────────
#  Forward pass custom — capture des activations
# ─────────────────────────────────────────────
def activate_with_hidden(net, inputs: list) -> tuple[list, dict]:
    """
    Forward pass identique à FeedForwardNetwork.activate()
    mais retourne aussi les activations des nœuds cachés.

    Retourne :
        outputs      : liste de 4 valeurs tanh (sorties)
        hidden_vals  : dict {node_key: activation_value}
    """
    output_set = set(net.output_nodes)
    values     = {k: v for k, v in zip(net.input_nodes, inputs)}

    hidden_vals = {}
    for node, act_func, agg_func, bias, response, links in net.node_evals:
        node_inputs = [values[i] * w for i, w in links]
        s           = agg_func(node_inputs)
        values[node] = act_func(bias + response * s)
        if node not in output_set:
            hidden_vals[node] = values[node]

    outputs = [values[k] for k in net.output_nodes]
    return outputs, hidden_vals


def _classify_situation(state: list, env: NeatSnakeEnv) -> int:
    """Classe l'état courant en une des 8 situations (distances en pixels bruts)."""
    DANGER_THR = 50   # 1 case en pixels
    d_n, d_e, d_s, d_w = state[0], state[2], state[4], state[6]
    food_h  = state[10] + state[14]   # food Est + food Ouest
    food_v  = state[8]  + state[12]   # food Nord + food Sud
    slen    = env.my_snake.lenght

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
                            fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=8)
    ax.tick_params(colors="#8899AA", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)


# ─────────────────────────────────────────────
#  Collecte des activations
# ─────────────────────────────────────────────
def collect_episodes(net, env: NeatSnakeEnv, n_episodes: int = 10):
    """
    Joue n_episodes épisodes, collecte activations hidden + métadonnées.
    Retourne :
        hidden_log    : list of dicts {node_key: value} per step
        node_keys     : sorted list of hidden node keys (consistent ordering)
        actions_log   : list of int
        situations_log: list of int
        scores_log    : list of int
    """
    hidden_log     = []
    actions_log    = []
    situations_log = []
    scores_log     = []
    node_keys      = None   # découverts au premier step

    for ep in range(n_episodes):
        state = env.reset()
        done  = False

        while not done:
            outputs, hidden = activate_with_hidden(net, state)
            action          = int(np.argmax(outputs))

            if node_keys is None and hidden:
                node_keys = sorted(hidden.keys())

            hidden_log.append(hidden)
            actions_log.append(action)
            situations_log.append(_classify_situation(state, env))
            scores_log.append(env.score)

            state, _, done, info = env.step(action)

        print(f"  [Collect] Épisode {ep+1}/{n_episodes} → "
              f"score {info['score']}  ({len(actions_log)} steps total)")

    if node_keys is None:
        node_keys = []

    return hidden_log, node_keys, actions_log, situations_log, scores_log


def build_act_matrix(hidden_log: list, node_keys: list) -> np.ndarray:
    """Retourne une matrice [T, N_hidden] à partir des logs."""
    if not node_keys:
        return np.zeros((len(hidden_log), 0))
    T = len(hidden_log)
    N = len(node_keys)
    mat = np.zeros((T, N), dtype=np.float32)
    for t, h in enumerate(hidden_log):
        for j, k in enumerate(node_keys):
            mat[t, j] = h.get(k, 0.0)
    return mat


# ─────────────────────────────────────────────
#  Analyse 1 — Distribution des activations
# ─────────────────────────────────────────────
def plot_distribution(acts: np.ndarray, node_keys: list):
    """
    Pour les nœuds cachés :
      - Histogramme global des valeurs tanh
      - Neurones saturés (|act| > 0.95 sur > 80% des steps)
      - Heatmap temporelle
    """
    T, N = acts.shape
    if N == 0:
        print("[XAI] Aucun nœud caché détecté — analyse de distribution ignorée.")
        return

    SAT_THR  = 0.95   # seuil de saturation tanh
    SAT_FRAC = 0.80   # saturé si > 80% du temps

    frac_sat = (np.abs(acts) > SAT_THR).mean(axis=0)   # [N]
    is_sat   = frac_sat > SAT_FRAC
    n_sat    = is_sat.sum()
    pct_sat  = 100 * n_sat / N

    fig = plt.figure(figsize=(22, 7), facecolor=BG)
    fig.suptitle(
        "Distribution des activations tanh — Nœuds cachés NEAT\n"
        "Neurones saturés = |tanh| > 0.95 sur > 80% des steps  |  "
        "Heatmap = activité temporelle (foncé=0, chaud=actif)",
        fontsize=13, fontweight="bold", color="white", y=1.01
    )

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38,
                           width_ratios=[1.4, 1, 2])

    # ── Col 0 : histogramme ───────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor(PANEL_BG)

    vals_flat = acts.flatten()
    ax0.hist(vals_flat, bins=80, color="#1F618D", alpha=0.65,
             edgecolor="none", label="Toutes")
    # Superposer les valeurs non-saturées
    mid = vals_flat[(np.abs(vals_flat) < SAT_THR)]
    if len(mid):
        ax0.hist(mid, bins=80, color="#F39C12", alpha=0.75,
                 edgecolor="none", label=f"|tanh| < {SAT_THR}")

    ax0.axvline(x=0,         color="#AAAAAA", linewidth=1.0,
                linestyle=":", alpha=0.5)
    ax0.axvline(x=SAT_THR,   color="#E74C3C", linewidth=1.3,
                linestyle="--", label=f"Saturation (+{SAT_THR})")
    ax0.axvline(x=-SAT_THR,  color="#E74C3C", linewidth=1.3,
                linestyle="--", label=f"Saturation (−{SAT_THR})")
    ax0.set_yscale("log")
    apply_style(ax0,
                title="Distribution des activations tanh\n(tous les nœuds cachés)",
                xlabel="Valeur tanh", ylabel="Fréquence (log)")
    ax0.legend(fontsize=7, facecolor="#0D1117",
               edgecolor="#444", labelcolor="white")

    stats_txt = (
        f"min  = {acts.min():.3f}\n"
        f"max  = {acts.max():.3f}\n"
        f"mean = {acts.mean():.3f}\n"
        f"std  = {acts.std():.3f}\n"
        f"saturés = {n_sat}/{N} ({pct_sat:.1f}%)"
    )
    ax0.text(0.97, 0.97, stats_txt,
             transform=ax0.transAxes, va="top", ha="right",
             color=TEXT_COL, fontsize=7,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#0D1B2A",
                       edgecolor=GRID_COL, alpha=0.9))

    # ── Col 1 : barplot des nœuds saturés ────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor(PANEL_BG)

    order   = np.argsort(frac_sat)[::-1]
    top_n   = min(40, N)
    bar_colors = [CMAP_SAT(frac_sat[order[i]]) for i in range(top_n)]

    ax1.barh(range(top_n), frac_sat[order[:top_n]],
             color=bar_colors, edgecolor="none", height=0.85)
    ax1.axvline(x=SAT_FRAC, color="#E74C3C", linewidth=1.2,
                linestyle="--", alpha=0.9, label=f"Seuil saturé ({int(SAT_FRAC*100)}%)")
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(
        [f"N{node_keys[order[i]]}" for i in range(top_n)],
        color=TEXT_COL, fontsize=7
    )
    ax1.set_xlim(0, 1.05)

    apply_style(ax1,
                title=f"Nœuds saturés (top {top_n})\n"
                       f"{n_sat}/{N} saturés ({pct_sat:.1f}%)",
                xlabel="Fraction de steps à |tanh| > 0.95")
    ax1.legend(fontsize=7, facecolor="#0D1117",
               edgecolor="#444", labelcolor="white")

    sm = ScalarMappable(cmap=CMAP_SAT, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Frac. saturée", color=TEXT_COL, fontsize=7)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=6)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    cbar.set_ticks([0, 0.5, SAT_FRAC, 1.0])
    cbar.set_ticklabels(["actif", "50%", "saturé", "100%"])

    # ── Col 2 : heatmap temporelle ───────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(PANEL_BG)

    t_show = min(200, T)
    n_show = min(80, N)
    var_order = np.argsort(acts.var(axis=0))[::-1][:n_show]
    heat_data = acts[:t_show, var_order].T

    vmax_h = np.percentile(np.abs(heat_data), 95)
    # Utiliser un colormap divergent centré sur 0 pour tanh
    im = ax2.imshow(
        heat_data,
        cmap="RdBu_r", vmin=-max(vmax_h, 1e-6), vmax=max(vmax_h, 1e-6),
        aspect="auto", interpolation="nearest"
    )
    ax2.set_xlabel("Step (temps)", color=TEXT_COL, fontsize=8)
    ax2.set_ylabel(f"Nœud (top {n_show} par variance)",
                   color=TEXT_COL, fontsize=8)
    ax2.set_title(
        f"Activité temporelle — {n_show} nœuds × {t_show} steps\n"
        "(triés par variance décroissante | rouge=+1, bleu=−1, blanc=0)",
        color="white", fontsize=10, fontweight="bold", pad=8
    )
    ax2.set_yticks(range(min(n_show, len(var_order))))
    ax2.set_yticklabels(
        [f"N{node_keys[var_order[i]]}" for i in range(min(n_show, len(var_order)))],
        color=TEXT_COL, fontsize=6
    )
    ax2.tick_params(colors="#8899AA", labelsize=7)
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRID_COL)

    cbar2 = plt.colorbar(im, ax=ax2, fraction=0.025, pad=0.02)
    cbar2.set_label("Activation tanh", color=TEXT_COL, fontsize=7)
    cbar2.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=6)
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color=TEXT_COL)

    plt.savefig(out("xai_neat_distribution.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out('xai_neat_distribution.png')}")
    plt.show()


# ─────────────────────────────────────────────
#  Analyse 2 — Neurones spécialisés
# ─────────────────────────────────────────────
def compute_specialization(acts: np.ndarray, situations: list,
                            node_keys: list) -> np.ndarray:
    """Retourne [N_nodes, 8] : activation moyenne par nœud × situation."""
    situations_arr = np.array(situations)
    N = acts.shape[1]
    S = len(SITUATION_NAMES)
    means = np.zeros((N, S))
    for si in range(S):
        mask = situations_arr == si
        if mask.sum() > 0:
            means[:, si] = acts[mask].mean(axis=0)
    return means


def plot_specialization(means: np.ndarray, acts: np.ndarray,
                        situations: list, node_keys: list):
    situations_arr = np.array(situations)
    sit_counts = [(situations_arr == si).sum() for si in range(len(SITUATION_NAMES))]
    N = means.shape[0]
    if N == 0:
        print("[XAI] Aucun nœud caché — spécialisation ignorée.")
        return

    # Score de spécialisation : max − mean sur les situations
    spec_score = means.max(axis=1) - means.mean(axis=1)
    top_idx    = np.argsort(spec_score)[::-1]

    fig = plt.figure(figsize=(24, 9), facecolor=BG)
    fig.suptitle(
        "Nœuds cachés NEAT spécialisés — Quels nœuds s'activent dans quelles situations ?\n"
        "Score = max_situation(act_moy) − mean_situations(act_moy)  |  Score élevé = nœud très sélectif",
        fontsize=12, fontweight="bold", color="white", y=1.005
    )

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42,
                           width_ratios=[1.2, 2, 1.8])

    # ── Col 0 : distribution des scores ──────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor(PANEL_BG)

    ax0.hist(spec_score, bins=min(40, N), color="#1F618D",
             alpha=0.7, edgecolor="none")
    ax0.axvline(x=np.percentile(spec_score, 90),
                color="#E74C3C", linewidth=1.5, linestyle="--",
                label="90e percentile")
    ax0.axvline(x=np.percentile(spec_score, 50),
                color="#F39C12", linewidth=1.0, linestyle=":",
                label="médiane")

    apply_style(ax0,
                title="Score de spécialisation\npar nœud caché NEAT",
                xlabel="max − mean des activations par situation",
                ylabel="Nombre de nœuds")
    ax0.legend(fontsize=7, facecolor="#0D1117",
               edgecolor="#444", labelcolor="white")

    top_node = node_keys[top_idx[0]] if node_keys else "?"
    ax0.text(0.97, 0.97,
             f"Top nœud : N{top_node}\n"
             f"Score max : {spec_score[top_idx[0]]:.3f}\n"
             f"Situation : {SITUATION_NAMES[means[top_idx[0]].argmax()].replace(chr(10),' ')}",
             transform=ax0.transAxes, va="top", ha="right",
             color="#FFD700", fontsize=7.5,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#0D1B2A",
                       edgecolor="#F39C12", alpha=0.9))

    # ── Col 1 : heatmap [situation × top-40 nœuds] ─
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor(PANEL_BG)

    top40  = top_idx[:min(40, N)]
    heat   = means[top40, :].T
    vmax_h = np.percentile(np.abs(means), 95)
    im = ax1.imshow(heat, cmap=CMAP_SPEC,
                    vmin=0, vmax=max(vmax_h, 1e-6),
                    aspect="auto", interpolation="nearest")

    ax1.set_yticks(range(len(SITUATION_NAMES)))
    ax1.set_yticklabels(
        [f"{n}  (n={sit_counts[i]})" for i, n in enumerate(SITUATION_NAMES)],
        color=TEXT_COL, fontsize=8
    )
    ax1.set_xticks(range(len(top40)))
    ax1.set_xticklabels(
        [f"N{node_keys[top40[i]]}" for i in range(len(top40))],
        rotation=90, color=TEXT_COL, fontsize=6.5
    )
    ax1.set_xlabel("Nœud (top 40 les plus spécialisés)",
                   color=TEXT_COL, fontsize=8)
    ax1.set_title(
        "Activation moyenne par situation × nœud\n"
        "(triés par score de spécialisation décroissant)",
        color="white", fontsize=10, fontweight="bold", pad=8
    )
    ax1.tick_params(colors="#8899AA", labelsize=7)
    for spine in ax1.spines.values():
        spine.set_edgecolor(GRID_COL)

    cbar1 = plt.colorbar(im, ax=ax1, fraction=0.03, pad=0.03)
    cbar1.set_label("Activation moy.", color=TEXT_COL, fontsize=7)
    cbar1.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=6)
    plt.setp(cbar1.ax.yaxis.get_ticklabels(), color=TEXT_COL)

    # ── Col 2 : profil des 5 nœuds les + spécialisés ─
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(PANEL_BG)

    top5    = top_idx[:min(5, N)]
    sit_pos = np.arange(len(SITUATION_NAMES))
    bar_w   = 0.15
    palette = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292", "#CE93D8"]

    for rank, nidx in enumerate(top5):
        offset = (rank - 2) * bar_w
        node_lbl = node_keys[nidx] if node_keys else nidx
        ax2.bar(sit_pos + offset, means[nidx],
                width=bar_w * 0.9,
                color=palette[rank], alpha=0.85,
                label=f"N{node_lbl} (score={spec_score[nidx]:.2f})",
                edgecolor="#0D1117")

    ax2.set_xticks(sit_pos)
    ax2.set_xticklabels(SITUATION_NAMES, rotation=35,
                        ha="right", color=TEXT_COL, fontsize=7.5)
    ax2.legend(fontsize=7, facecolor="#0D1117",
               edgecolor="#444", labelcolor="white",
               loc="upper right")
    apply_style(ax2,
                title="Profil des 5 nœuds les plus spécialisés",
                ylabel="Activation moyenne (tanh)")
    ax2.grid(axis="y", color=GRID_COL, linewidth=0.5, alpha=0.5)

    plt.savefig(out("xai_neat_specialization.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out('xai_neat_specialization.png')}")
    plt.show()


# ─────────────────────────────────────────────
#  Analyse 3 — t-SNE / UMAP des activations
# ─────────────────────────────────────────────
def _run_tsne(data: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    from sklearn.manifold import TSNE
    import sklearn
    from packaging import version

    iter_kwarg = (
        "max_iter"
        if version.parse(sklearn.__version__) >= version.parse("1.4")
        else "n_iter"
    )
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, data.shape[0] - 1),
        learning_rate="auto",
        init="random" if data.shape[1] < 2 else "pca",
        random_state=42,
        **{iter_kwarg: 1000},
    )
    return tsne.fit_transform(data)


def _run_umap(data: np.ndarray) -> np.ndarray:
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=15,
                            min_dist=0.1, random_state=42)
        return reducer.fit_transform(data)
    except ImportError:
        print("  [WARN] umap-learn non installé → fallback t-SNE.")
        return _run_tsne(data)


def plot_projection(acts: np.ndarray, situations: list,
                    actions: list, scores: list,
                    node_keys: list, method: str = "tsne"):
    if acts.shape[1] == 0:
        print("[XAI] Aucun nœud caché — projection ignorée.")
        return
    if acts.shape[1] < 2:
        print(f"[XAI] Seulement {acts.shape[1]} nœud(s) caché(s) — "
              "projection 2D impossible (minimum 2 dimensions requises). "
              "Analyse ignorée.")
        return

    method_label = "t-SNE" if method == "tsne" else "UMAP"
    situations_arr = np.array(situations)
    actions_arr    = np.array(actions)
    scores_arr     = np.array(scores, dtype=float)

    MAX_POINTS = 3000
    T_total    = len(situations)
    idx = np.random.choice(T_total, min(MAX_POINTS, T_total), replace=False)
    idx = np.sort(idx)

    acts_sub      = acts[idx]
    sits_sub      = situations_arr[idx]
    actions_sub   = actions_arr[idx]
    scores_sub    = scores_arr[idx]

    print(f"  [{method_label}] {acts_sub.shape[0]} points × "
          f"{acts_sub.shape[1]} nœuds…")

    if method == "tsne":
        proj = _run_tsne(acts_sub)
    else:
        proj = _run_umap(acts_sub)

    x, y  = proj[:, 0], proj[:, 1]
    ALPHA = 0.55
    SIZE  = 8

    sit_colors_map = {
        0: "#E74C3C", 1: "#F39C12", 2: "#F1C40F", 3: "#2ECC71",
        4: "#3498DB", 5: "#9B59B6", 6: "#1ABC9C", 7: "#95A5A6",
    }

    from matplotlib.colors import LinearSegmentedColormap as LSC
    CMAP_SCORE = LSC.from_list(
        "score", ["#0D1B2A", "#1F618D", "#2ECC71", "#F39C12", "#E74C3C"]
    )

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor=BG)
    fig.suptitle(
        f"Projection {method_label} des activations nœuds cachés NEAT\n"
        "Gauche : situation  |  Centre : action choisie  |  Droite : score",
        fontsize=13, fontweight="bold", color="white"
    )

    def _style_ax(ax, title):
        ax.set_facecolor(PANEL_BG)
        ax.set_title(title, color="white", fontsize=11,
                     fontweight="bold", pad=8)
        ax.set_xlabel(f"{method_label}-1", color=TEXT_COL, fontsize=8)
        ax.set_ylabel(f"{method_label}-2", color=TEXT_COL, fontsize=8)
        ax.tick_params(colors="#8899AA", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)

    # ── Gauche : par situation ────────────────────
    ax0 = axes[0]
    for si, sname in enumerate(SITUATION_NAMES):
        mask = sits_sub == si
        if not mask.any(): continue
        ax0.scatter(x[mask], y[mask], c=sit_colors_map[si],
                    s=SIZE, alpha=ALPHA, edgecolors="none",
                    label=f"{sname} ({mask.sum()})")
    ax0.legend(fontsize=6.5, facecolor="#0D1117", edgecolor="#444",
               labelcolor="white", markerscale=2, loc="best", framealpha=0.85)
    _style_ax(ax0, f"{method_label} — Coloré par situation")

    # ── Centre : par action ───────────────────────
    ax1 = axes[1]
    for ai, aname in enumerate(ACTION_NAMES):
        mask = actions_sub == ai
        if not mask.any(): continue
        ax1.scatter(x[mask], y[mask], c=ACTION_COLORS[ai],
                    s=SIZE, alpha=ALPHA, edgecolors="none",
                    label=f"{aname} ({mask.sum()})")
    ax1.legend(fontsize=7, facecolor="#0D1117", edgecolor="#444",
               labelcolor="white", markerscale=2, loc="best", framealpha=0.85)
    _style_ax(ax1, f"{method_label} — Coloré par action choisie")

    # ── Droite : par score ────────────────────────
    ax2 = axes[2]
    sc = ax2.scatter(x, y, c=scores_sub, cmap=CMAP_SCORE,
                     s=SIZE, alpha=ALPHA, edgecolors="none",
                     vmin=scores_sub.min(),
                     vmax=max(scores_sub.max(), 1))
    cbar = plt.colorbar(sc, ax=ax2, fraction=0.04, pad=0.03)
    cbar.set_label("Score courant", color=TEXT_COL, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    _style_ax(ax2, f"{method_label} — Coloré par score")

    plt.tight_layout()
    fname = f"xai_neat_{method}.png"
    plt.savefig(out(fname), dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out(fname)}")
    plt.show()


# ─────────────────────────────────────────────
#  Point d'entrée
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="XAI — Activations NEAT Snake"
    )
    parser.add_argument("--distribution",   action="store_true")
    parser.add_argument("--specialization", action="store_true")
    parser.add_argument("--tsne",           action="store_true")
    parser.add_argument("--umap",           action="store_true")
    parser.add_argument("--model",    type=str, default="best_model_11")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    run_all = not (args.distribution or args.specialization
                   or args.tsne or args.umap)

    net, genome, config = load_neat_model(args.model)
    env = NeatSnakeEnv()

    n_hidden = sum(
        1 for k in genome.nodes
        if k not in config.genome_config.input_keys
        and k not in config.genome_config.output_keys
    )
    print(f"[XAI] Modèle : {args.model}.pkl")
    print(f"[XAI] Nœuds cachés : {n_hidden} | "
          f"Sorties : {len(config.genome_config.output_keys)}")

    print(f"\n[XAI] Collecte sur {args.episodes} épisode(s)…")
    hidden_log, node_keys, actions, situations, scores = collect_episodes(
        net, env, n_episodes=args.episodes
    )
    acts = build_act_matrix(hidden_log, node_keys)
    print(f"[XAI] {len(actions)} steps | "
          f"{len(node_keys)} nœuds cachés détectés.\n")

    if run_all or args.distribution:
        print("[XAI] ── Distribution des activations ───────────────")
        plot_distribution(acts, node_keys)

    if run_all or args.specialization:
        print("[XAI] ── Nœuds spécialisés ──────────────────────────")
        means = compute_specialization(acts, situations, node_keys)
        plot_specialization(means, acts, situations, node_keys)

    if run_all or args.tsne:
        print("[XAI] ── t-SNE des activations ──────────────────────")
        try:
            from sklearn.manifold import TSNE
            plot_projection(acts, situations, actions, scores,
                            node_keys, method="tsne")
        except ImportError:
            print("  [WARN] scikit-learn non installé.")

    if run_all or args.umap:
        print("[XAI] ── UMAP des activations ───────────────────────")
        plot_projection(acts, situations, actions, scores,
                        node_keys, method="umap")

    print("\n[XAI] Analyse terminée.")


if __name__ == "__main__":
    main()

# python xai_neat_activations.py                         # tout (10 épisodes)
# python xai_neat_activations.py --distribution          # rapide, pas de sklearn
# python xai_neat_activations.py --specialization
# python xai_neat_activations.py --tsne --episodes 20
# python xai_neat_activations.py --umap --episodes 20
# python xai_neat_activations.py --model best_model_8
