"""
xai_neat_outputs.py — Analyse XAI : Sorties du réseau NEAT Snake
=================================================================
Équivalent de xai_qvalues.py pour l'algorithme génétique NEAT.
Les sorties NEAT sont des valeurs tanh ∈ [-1, 1] (non des Q-values).
L'action choisie = argmax des 4 sorties.

3 visualisations :
  1. Heatmaps des 4 sorties sur la grille (nourriture fixe)
  2. Carte de confiance (gap max − 2e max) + politique apprise
  3. Évolution temporelle des sorties pendant un épisode

Usage :
    python xai_neat_outputs.py                   # tout
    python xai_neat_outputs.py --heatmap         # heatmaps des sorties
    python xai_neat_outputs.py --gap             # confiance + politique
    python xai_neat_outputs.py --temporal        # évolution temporelle
    python xai_neat_outputs.py --model best_model_11   # modèle (défaut)
    python xai_neat_outputs.py --food-col 8 --food-row 4
    python xai_neat_outputs.py --episodes 3
"""

import argparse
import math
import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

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
OUT_DIR = "xai_neat_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def out(filename: str) -> str:
    return os.path.join(OUT_DIR, filename)


# ─────────────────────────────────────────────
#  Constantes & palettes
# ─────────────────────────────────────────────
ACTION_NAMES  = ["UP ↑", "RIGHT →", "DOWN ↓", "LEFT ←"]
ACTION_COLORS = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"]

GRID_W = int(game.width  // game.rect_width)    # 16 colonnes
GRID_H = int(game.height // game.rect_height)   # 8 lignes

BG       = "#0D1117"
PANEL_BG = "#0D1B2A"
GRID_COL = "#1E3A5F"
TEXT_COL = "#CCDDEE"

# Colormap tanh : bleu froid (−1) → blanc (0) → rouge chaud (+1)
CMAP_OUT = LinearSegmentedColormap.from_list(
    "neat_out", ["#0D1B2A", "#1B4F72", "#2E86C1", "#F39C12", "#E74C3C", "#FFFFFF"]
)
CMAP_GAP = LinearSegmentedColormap.from_list(
    "gap", ["#1A1A2E", "#16213E", "#0F3460", "#533483", "#E94560"]
)


# ─────────────────────────────────────────────
#  Environnement Snake pour NEAT (step-by-step)
# ─────────────────────────────────────────────
class NeatSnakeEnv:
    """Environnement Snake pas-à-pas compatible avec l'API NEAT."""

    def reset(self):
        self.my_snake = game.Manager_snake()
        self.my_snake.add_snake(game.Snake(5 * game.rect_width,
                                           5 * game.rect_height))
        self.food      = game.generated_food(self.my_snake)
        self.score     = 0
        self.iteration = 0
        return self._get_state()

    def _get_state(self):
        s = self.my_snake
        f = self.food
        return [
            game.distance_bord_north(s),
            game.distance_bord_north_est(s),
            game.distance_bord_est(s),
            game.distance_bord_south_est(s),
            game.distance_bord_south(s),
            game.distance_bord_south_west(s),
            game.distance_bord_west(s),
            game.distance_bord_north_west(s),
            game.distance_food_north(s, f),
            game.distance_food_north_est(s, f),
            game.distance_food_est(s, f),
            game.distance_food_south_est(s, f),
            game.distance_food_south(s, f),
            game.distance_food_south_west(s, f),
            game.distance_food_west(s, f),
            game.distance_food_north_west(s, f),
        ]

    def step(self, action):
        # 1. Mise à jour de la direction (règle : pas de demi-tour)
        if   action == 0 and self.my_snake.direction != "DOWN":
            self.my_snake.direction = "UP"
        elif action == 2 and self.my_snake.direction != "UP":
            self.my_snake.direction = "DOWN"
        elif action == 1 and self.my_snake.direction != "LEFT":
            self.my_snake.direction = "RIGHT"
        elif action == 3 and self.my_snake.direction != "RIGHT":
            self.my_snake.direction = "LEFT"

        # 2. Nourriture à la position actuelle de la tête
        head = self.my_snake.list_snake[0]
        if head.x == self.food.x and head.y == self.food.y:
            tail = self.my_snake.list_snake[-1]
            self.my_snake.add_snake(game.Snake(tail.x, tail.y))
            self.food   = game.generated_food(self.my_snake)
            self.score += 1

        # 3. Déplacement
        alive          = self.my_snake.move()
        self.iteration += 1
        done           = not alive or self.iteration >= game.stop_iteration
        return self._get_state(), float(alive) - 1.0, done, {
            "score": self.score, "iteration": self.iteration
        }


# ─────────────────────────────────────────────
#  Chargement du modèle NEAT
# ─────────────────────────────────────────────
def load_neat_model(model_name: str = "best_model_11"):
    config_path = os.path.join(os.path.dirname(__file__), "config.txt")
    config = neat.config.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path
    )
    pkl_path = f"{model_name}.pkl"
    try:
        with open(pkl_path, "rb") as f:
            genome = pickle.load(f)
    except FileNotFoundError:
        print(f"[WARN] {pkl_path} introuvable — génome aléatoire.")
        genome = None

    if genome is not None:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
    else:
        pop = neat.Population(config)
        genome = list(pop.population.values())[0]
        net    = neat.nn.FeedForwardNetwork.create(genome, config)

    return net, genome, config


# ─────────────────────────────────────────────
#  Utilitaires
# ─────────────────────────────────────────────
def get_outputs(net, state: list) -> np.ndarray:
    """Retourne les 4 sorties NEAT (tanh) pour un état donné."""
    return np.array(net.activate(state), dtype=np.float32)


def build_state_at(col: int, row: int, food_col: int, food_row: int) -> list:
    """
    Construit l'état (distances brutes en pixels) pour une tête
    en (col, row) et une nourriture en (food_col, food_row).
    Serpent à 1 seul segment (tête seulement).
    """
    tmp_snake = game.Manager_snake()
    tmp_snake.add_snake(game.Snake(col * game.rect_width,
                                   row * game.rect_height))
    tmp_food = game.food(food_col * game.rect_width,
                         food_row * game.rect_height)
    return [
        game.distance_bord_north(tmp_snake),
        game.distance_bord_north_est(tmp_snake),
        game.distance_bord_est(tmp_snake),
        game.distance_bord_south_est(tmp_snake),
        game.distance_bord_south(tmp_snake),
        game.distance_bord_south_west(tmp_snake),
        game.distance_bord_west(tmp_snake),
        game.distance_bord_north_west(tmp_snake),
        game.distance_food_north(tmp_snake, tmp_food),
        game.distance_food_north_est(tmp_snake, tmp_food),
        game.distance_food_est(tmp_snake, tmp_food),
        game.distance_food_south_est(tmp_snake, tmp_food),
        game.distance_food_south(tmp_snake, tmp_food),
        game.distance_food_south_west(tmp_snake, tmp_food),
        game.distance_food_west(tmp_snake, tmp_food),
        game.distance_food_north_west(tmp_snake, tmp_food),
    ]


def scan_grid(net, food_col: int, food_row: int):
    """
    Parcourt toutes les cellules de la grille et calcule
    les sorties NEAT pour chaque position de tête.

    Retourne :
        out_map : [GRID_H, GRID_W, 4]  — sorties par action
        best    : [GRID_H, GRID_W]     — action choisie (argmax)
        gap     : [GRID_H, GRID_W]     — max − 2e max (confiance)
    """
    out_map = np.zeros((GRID_H, GRID_W, 4), dtype=np.float32)
    for row in range(GRID_H):
        for col in range(GRID_W):
            state          = build_state_at(col, row, food_col, food_row)
            out_map[row, col] = get_outputs(net, state)

    sorted_o = np.sort(out_map, axis=2)
    best     = np.argmax(out_map, axis=2)
    gap      = sorted_o[:, :, -1] - sorted_o[:, :, -2]
    return out_map, best, gap


# ─────────────────────────────────────────────
#  Visualisation 1 — Heatmaps des sorties
# ─────────────────────────────────────────────
def plot_output_heatmaps(net, food_col: int = 8, food_row: int = 4):
    out_map, best, gap = scan_grid(net, food_col, food_row)

    fig = plt.figure(figsize=(22, 7), facecolor=BG)
    fig.suptitle(
        "Sorties NEAT par action — position de la nourriture fixée\n"
        "(valeurs tanh ∈ [−1, +1] ; argmax = action choisie)",
        fontsize=16, fontweight="bold", color="white", y=1.01
    )

    gs   = gridspec.GridSpec(1, 5, figure=fig, wspace=0.35)
    vmin = out_map.min()
    vmax = out_map.max()

    for i, aname in enumerate(ACTION_NAMES):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(BG)

        im = ax.imshow(
            out_map[:, :, i],
            cmap=CMAP_OUT, vmin=vmin, vmax=vmax,
            interpolation="nearest", aspect="auto"
        )
        ax.scatter(food_col, food_row, marker="*", s=350,
                   color="#FFD700", zorder=5, label="Nourriture")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        ax.set_title(aname, color=ACTION_COLORS[i],
                     fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Colonne", color=TEXT_COL, fontsize=8)
        ax.set_ylabel("Ligne",   color=TEXT_COL, fontsize=8)
        ax.tick_params(colors="#888888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)

    plt.tight_layout()
    plt.savefig(out("xai_neat_heatmaps.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out('xai_neat_heatmaps.png')}")
    plt.show()


# ─────────────────────────────────────────────
#  Visualisation 2 — Confiance + politique
# ─────────────────────────────────────────────
def plot_confidence_map(net, food_col: int = 8, food_row: int = 4):
    out_map, best, gap = scan_grid(net, food_col, food_row)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
    fig.suptitle(
        "Confiance de l'agent NEAT (gap sorties) & Politique apprise",
        fontsize=15, fontweight="bold", color="white"
    )

    # ── Gauche : heatmap du gap ─────────────────
    ax1 = axes[0]
    ax1.set_facecolor(BG)
    im = ax1.imshow(gap, cmap=CMAP_GAP,
                    interpolation="nearest", aspect="auto")
    ax1.scatter(food_col, food_row, marker="*", s=400,
                color="#FFD700", zorder=5)

    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Sortie_max − Sortie_2nd", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax1.set_title("Confiance (gap sorties tanh)", color="white",
                  fontsize=13, pad=10)
    ax1.set_xlabel("Colonne", color=TEXT_COL, fontsize=9)
    ax1.set_ylabel("Ligne",   color=TEXT_COL, fontsize=9)
    ax1.tick_params(colors="#888888", labelsize=7)
    for spine in ax1.spines.values():
        spine.set_edgecolor(GRID_COL)

    # ── Droite : politique (action choisie) ────────
    ax2 = axes[1]
    ax2.set_facecolor(BG)

    color_table = {
        0: np.array([0.31, 0.76, 0.97]),
        1: np.array([0.51, 0.78, 0.52]),
        2: np.array([1.00, 0.72, 0.30]),
        3: np.array([0.94, 0.38, 0.57]),
    }
    policy_rgb = np.zeros((GRID_H, GRID_W, 3))
    for r in range(GRID_H):
        for c in range(GRID_W):
            policy_rgb[r, c] = color_table[best[r, c]]

    gap_norm = (gap - gap.min()) / (gap.max() - gap.min() + 1e-8)
    alpha    = 0.35 + 0.65 * gap_norm
    for ch in range(3):
        policy_rgb[:, :, ch] *= alpha

    ax2.imshow(policy_rgb, interpolation="nearest", aspect="auto")

    arrows = {0: (0, -0.35), 1: (0.35, 0), 2: (0, 0.35), 3: (-0.35, 0)}
    for r in range(GRID_H):
        for c in range(GRID_W):
            dx, dy = arrows[best[r, c]]
            ax2.annotate(
                "", xy=(c + dx, r + dy), xytext=(c, r),
                arrowprops=dict(arrowstyle="->", color="white", lw=0.8),
            )

    ax2.scatter(food_col, food_row, marker="*", s=400,
                color="#FFD700", zorder=5)

    legend_patches = [
        mpatches.Patch(color=tuple(color_table[i]), label=ACTION_NAMES[i])
        for i in range(4)
    ]
    ax2.legend(handles=legend_patches, loc="upper right",
               fontsize=8, facecolor="#1A1A2E", edgecolor="#444",
               labelcolor="white")
    ax2.set_title("Politique apprise (action argmax)", color="white",
                  fontsize=13, pad=10)
    ax2.set_xlabel("Colonne", color=TEXT_COL, fontsize=9)
    ax2.set_ylabel("Ligne",   color=TEXT_COL, fontsize=9)
    ax2.tick_params(colors="#888888", labelsize=7)
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRID_COL)

    plt.tight_layout()
    plt.savefig(out("xai_neat_confidence.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out('xai_neat_confidence.png')}")
    plt.show()


# ─────────────────────────────────────────────
#  Visualisation 3 — Évolution temporelle
# ─────────────────────────────────────────────
def plot_temporal_outputs(net, num_episodes: int = 3):
    """
    Lance N épisodes en greedy et enregistre les 4 sorties à chaque step.
    Affiche l'évolution temporelle + événements (nourriture / mort).
    """
    env         = NeatSnakeEnv()
    all_episodes = []

    for ep in range(num_episodes):
        state   = env.reset()
        done    = False
        ep_data = {"outputs": [], "events": [], "scores": []}
        step    = 0
        prev_sc = 0

        while not done:
            outs   = get_outputs(net, state)
            action = int(np.argmax(outs))
            ep_data["outputs"].append(outs.copy())

            state, _, done, info = env.step(action)

            if info["score"] > prev_sc:
                ep_data["events"].append((step, "food"))
                prev_sc = info["score"]
            if done and info["iteration"] < game.stop_iteration:
                ep_data["events"].append((step, "death"))

            ep_data["scores"].append(info["score"])
            step += 1

        all_episodes.append(ep_data)
        print(f"[XAI] Épisode {ep+1} → Score : {info['score']}  ({step} steps)")

    fig, axes = plt.subplots(
        num_episodes, 1,
        figsize=(18, 5 * num_episodes),
        facecolor=BG, squeeze=False
    )
    fig.suptitle(
        "Évolution temporelle des sorties NEAT pendant l'épisode\n"
        "Valeurs tanh ∈ [−1, +1]  |  vert pointillé = nourriture  |  rouge = mort",
        fontsize=14, fontweight="bold", color="white", y=1.01
    )

    for ep_idx, ep_data in enumerate(all_episodes):
        ax      = axes[ep_idx, 0]
        ax.set_facecolor(PANEL_BG)
        outs_arr = np.array(ep_data["outputs"])
        T        = len(outs_arr)
        steps    = np.arange(T)

        for i in range(4):
            ax.fill_between(steps, outs_arr[:, i],
                            alpha=0.08, color=ACTION_COLORS[i])
            ax.plot(steps, outs_arr[:, i],
                    label=ACTION_NAMES[i], color=ACTION_COLORS[i],
                    linewidth=1.4, alpha=0.9)

        ax.plot(steps, outs_arr.max(axis=1),
                color="white", linewidth=0.8, linestyle="--",
                alpha=0.5, label="Max sortie")
        ax.axhline(y=0, color="#555577", linewidth=0.7,
                   linestyle=":", alpha=0.6)

        for step_ev, ev_type in ep_data["events"]:
            if ev_type == "food":
                ax.axvline(x=step_ev, color="#2ECC71",
                           linewidth=1.5, linestyle=":", alpha=0.8)
                ax.text(step_ev + 0.5, ax.get_ylim()[1] * 0.9,
                        "🍎", fontsize=10, color="#2ECC71", alpha=0.9)
            elif ev_type == "death":
                ax.axvline(x=step_ev, color="#E74C3C",
                           linewidth=2.0, linestyle="-", alpha=0.9)
                ax.text(step_ev + 0.5, ax.get_ylim()[1] * 0.9,
                        "💀", fontsize=10, color="#E74C3C")

        score_final = ep_data["scores"][-1] if ep_data["scores"] else 0
        ax.set_title(
            f"Épisode {ep_idx+1}  —  Score : {score_final}  |  Steps : {T}",
            color="white", fontsize=12, pad=8
        )
        ax.set_xlabel("Step", color=TEXT_COL, fontsize=9)
        ax.set_ylabel("Sortie tanh", color=TEXT_COL, fontsize=9)
        ax.set_ylim(-1.1, 1.1)
        ax.tick_params(colors="#888888", labelsize=8)
        ax.legend(loc="upper left", fontsize=8, facecolor="#0D1117",
                  edgecolor="#444444", labelcolor="white",
                  framealpha=0.8, ncol=5)
        ax.grid(axis="y", color=GRID_COL, linewidth=0.5, alpha=0.6)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)

    plt.tight_layout()
    plt.savefig(out("xai_neat_temporal.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out('xai_neat_temporal.png')}")
    plt.show()


# ─────────────────────────────────────────────
#  Point d'entrée
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="XAI — Sorties NEAT Snake"
    )
    parser.add_argument("--heatmap",  action="store_true")
    parser.add_argument("--gap",      action="store_true")
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--model",    type=str, default="best_model_11")
    parser.add_argument("--food-col", type=int, default=8)
    parser.add_argument("--food-row", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    run_all = not (args.heatmap or args.gap or args.temporal)
    net, genome, config = load_neat_model(args.model)
    print(f"[XAI] Modèle chargé : {args.model}.pkl")
    print(f"[XAI] Nœuds : {len(genome.nodes)} | "
          f"Connexions actives : "
          f"{sum(1 for c in genome.connections.values() if c.enabled)}")

    if run_all or args.heatmap:
        print("\n[XAI] ── Heatmaps des sorties ──────────────────────")
        plot_output_heatmaps(net, args.food_col, args.food_row)

    if run_all or args.gap:
        print("\n[XAI] ── Carte de confiance + politique ────────────")
        plot_confidence_map(net, args.food_col, args.food_row)

    if run_all or args.temporal:
        print(f"\n[XAI] ── Évolution temporelle ({args.episodes} épisode(s)) ──")
        plot_temporal_outputs(net, args.episodes)

    print("\n[XAI] Analyse terminée.")


if __name__ == "__main__":
    main()

# python xai_neat_outputs.py                         # tout
# python xai_neat_outputs.py --heatmap
# python xai_neat_outputs.py --gap
# python xai_neat_outputs.py --temporal --episodes 5
# python xai_neat_outputs.py --food-col 12 --food-row 2
# python xai_neat_outputs.py --model best_model_8
