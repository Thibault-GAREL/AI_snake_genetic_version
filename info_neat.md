# Snake NEAT — Informations d'entraînement

## Résumé XAI — Interprétabilité du modèle

L'analyse XAI du réseau NEAT révèle une stratégie émergente claire et cohérente. Les analyses SHAP et de permutation s'accordent : **les distances à la nourriture** (`Dist_food_*`) sont les features les plus influentes, en particulier dans la direction cible de chaque action, tandis que les distances aux murs jouent un rôle secondaire d'évitement. La corrélation features→actions confirme ce comportement directionnel : l'agent oriente sa décision vers la direction où la nourriture est la plus proche.

Du côté des neurones cachés, un nœud est saturé en permanence (tanh ≈ 1, 100% des steps), agissant comme un biais appris, tandis que les autres nœuds présentent une spécialisation situationnelle modérée (danger directionnel, nourriture alignée). Les heatmaps de sorties et l'analyse temporelle montrent que l'agent adopte une **politique binaire franche** : il alterne entre deux directions dominantes avec des transitions nettes à chaque collecte de nourriture, sans hésitation, mais au détriment d'une exploration plus globale de la grille.

Les waterfall SHAP par situation valident la cohérence interne du modèle : face à un danger, les distances aux murs déclenchent l'évitement ; face à de la nourriture alignée, la distance dans la bonne direction domine la décision. Ce comportement simple mais efficace explique à la fois les performances atteintes (> 20 pommes) et les limites du modèle face à des configurations plus complexes.

---

## Temps d'entraînement

| Paramètre | Valeur | Source |
|---|---|---|
| Durée totale estimée | ~15 heures | README.md |
| Nombre de runs d'entraînement | ~11 runs | (best_model_11.pkl = 11ème sauvegarde) |
| Durée par run (estimation) | ~1h à 1h30 | 21 générations × ~3–4 min/génération |
| Générations par run | 21 (`nb_loop_train = 20 + 1`) | ia.py ligne 16–18 |
| Itérations max par partie | 500 (`stop_iteration`) | snake.py ligne 14 |
| Vitesse affichage | 11 FPS (`vitesse = 11`) | snake.py ligne 16 |
| Référence de vitesse | ~6 000 itérations/min | commentaire ia.py ligne 16 |

**Estimation détaillée par génération :**
- 100 génomes × ~200 steps moyens = ~20 000 steps
- À ~6 000 steps/min → ~3,3 min/génération
- 21 générations → ~70 min/run
- 11 runs → ~12–15h au total ✓

---

## Architecture du réseau (neurones)

| Couche | Nombre | Détail |
|---|---|---|
| **Entrées** | **16** | 8 distances aux murs + 8 distances à la nourriture (8 directions chacune) |
| **Cachés (initial)** | **8** | Peut évoluer via mutations NEAT (ajout/suppression de nœuds) |
| **Sorties** | **4** | UP / RIGHT / DOWN / LEFT |
| **Total initial** | **28** | Feed-forward, connexions complètes (`full_direct`) |

### Détail des 16 entrées

| # | Capteur | Direction |
|---|---|---|
| 1–8 | Distance au mur | N, NE, E, SE, S, SW, W, NW |
| 9–16 | Distance à la nourriture | N, NE, E, SE, S, SW, W, NW |

### Évolution topologique (NEAT)

| Mutation | Probabilité |
|---|---|
| Ajout nœud | 20% (`node_add_prob = 0.2`) |
| Suppression nœud | 20% (`node_delete_prob = 0.2`) |
| Ajout connexion | 50% (`conn_add_prob = 0.5`) |
| Suppression connexion | 50% (`conn_delete_prob = 0.5`) |

- Fonction d'activation : **tanh** (fixe, pas de mutation)
- Poids : bornes [-30, +30], mutation rate 80%

---

## Mémoire / Population (équivalent batch)

> NEAT est un algorithme évolutionnaire : **pas de replay buffer ni de batch gradient**.
> L'équivalent "batch" est la **population de génomes** évaluée en parallèle.

| Paramètre | Valeur | Source |
|---|---|---|
| **Taille de population** | **100** génomes | config.txt ligne 5 (`pop_size`) |
| Survie par génération | Top 20% → **20 génomes** | `survival_threshold = 0.2` |
| Élitisme | 1 genome préservé exactement | `elitism = 1` |
| Stagnation max (espèces) | 100 générations | `max_stagnation = 100` |
| Seuil de compatibilité | 3.0 | `compatibility_threshold = 3.0` |

---

## Performance

| Métrique | Valeur | Source |
|---|---|---|
| **Score max atteint** | **> 20 pommes** | README.md |
| **Seuil d'arrêt (fitness)** | **30** | config.txt (`fitness_threshold = 30`) |
| Critère de fitness | `max` | config.txt (`fitness_criterion = max`) |
| Fitness = score | Oui — `genome.fitness = score` retourné par `game_loop()` | snake.py ligne 475, ia.py ligne 81 |
| Meilleur modèle | `best_model_11.pkl` | ia.py ligne 158 |

### Suivi de la fitness

- **Par génération** : fitness moyenne et meilleure fitness loggées dans `donnees_neat.xlsx`
- Variables trackées : `best_fitnesses[]` et `mean_fitnesses[]` (ia.py lignes 62–63)
- Format Excel : feuilles nommées `entrainement_N` (une par run)

---

## Environnement de jeu

| Paramètre | Valeur |
|---|---|
| Taille de la grille | 800 × 400 px |
| Taille d'une cellule | 50 × 50 px |
| Cases disponibles | 16 × 8 = **128 positions** |
| Position initiale du snake | (250, 250) |
