import neat
import os
import sys
import pickle
import numpy as np

from graphviz import Digraph

import snake

import exw
import compteur

input_dim = 16 #state dim : 8 bords, 8 foods

nb_loop_train = 20 # * 6000 # 6000 = 1 min

nb_loop_train += 1

def visualize_neat_network(genome, config):
    dot = Digraph(format='png', engine='dot')

    # Ajouter les nœuds d'entrée
    for node_id in config.genome_config.input_keys:
        dot.node(str(node_id), f"Entrée {node_id}", shape="box", style="filled", color="lightblue")

    # Ajouter les nœuds de sortie
    for node_id in config.genome_config.output_keys:
        dot.node(str(node_id), f"Sortie {node_id}", shape="box", style="filled", color="lightgreen")

    # Ajouter les nœuds cachés
    for node_id in genome.nodes:
        if node_id not in config.genome_config.input_keys and node_id not in config.genome_config.output_keys:
            dot.node(str(node_id), f"Caché {node_id}", shape="circle", style="filled", color="lightgray")

    # Ajouter les connexions
    for conn_key, conn_gene in genome.connections.items():
        source, target = conn_key
        if conn_gene.enabled:
            dot.edge(str(source), str(target), label=f"{conn_gene.weight:.2f}", color="black")
        else:
            dot.edge(str(source), str(target), label=f"{conn_gene.weight:.2f}", color="red", style="dotted")

    # Sauvegarder et afficher le graphe
    dot.render('network_graph', view=True)

class Neat:
    def __init__(self):
        self.initNeat()
    def initNeat(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, "config.txt")
        self.config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                         neat.DefaultStagnation, config_path)
        self.p = neat.Population(self.config)
        self.p.add_reporter(neat.StdOutReporter(True))
        self.reporter = neat.StatisticsReporter()
        self.p.add_reporter(self.reporter)

        # self.controlplayer = False

        self.best_fitnesses = []
        self.mean_fitnesses = []

        self.alpha = 0.0
        self.omega = 8.0  # Durée d'une game en secondes

        # self.displayCollisionNodes()

        # self.taskMgr.stop()
        # self.taskMgr.add(self.updateCamera, "Update Camera")
        # self.taskMgr.add(self.updatePlayerNeat, "Update Player")
        # self.taskMgr.add(self.updateCollisionsNeat, "Update Collisions")
        # self.taskMgr.add(self.verifFinPartie, "Vérification fin de Game")

    def eval_genomes(self, genomes, config):
        j = 0
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            # game = SnakeGame() # Ici c'est une itération qui renvoie (state // action // score // next_state // done) attention le next_state n'est pas pris en compte
            genome.fitness = snake.game_loop(snake.rect_width, snake.rect_height, snake.display, net, genome,
                                       j, Neat)  # Ici je pense que il faut juste return le score simplement après une game
            j += 1
            # while not game.done:
            #     state = game.get_state()
            #     action = get_action(net, state)
            #     _, done = game.step(action)
            #     genome.fitness += game.score
            #      # Mettre la même init + game_loop

    def get_action(net, state):
        return np.argmax(net.activate(state))

    def tab_state(distance_bord_n, distance_bord_n_e, distance_bord_e, distance_bord_s_e, distance_bord_s,
                  distance_bord_s_w, distance_bord_w, distance_bord_n_w, distance_food_n, distance_food_n_e,
                  distance_food_e, distance_food_s_e, distance_food_s, distance_food_s_w, distance_food_w,
                  distance_food_n_w):
        return [distance_bord_n, distance_bord_n_e, distance_bord_e, distance_bord_s_e, distance_bord_s, distance_bord_s_w,
             distance_bord_w, distance_bord_n_w, distance_food_n, distance_food_n_e, distance_food_e, distance_food_s_e,
             distance_food_s, distance_food_s_w, distance_food_w, distance_food_n_w]



    def runNeat(self, nb_loop_train):
        # stats = neat.StatisticsReporter()
        # self.p.add_reporter(stats)
        #
        # winner = self.p.run(self.eval_genomes, nb_loop_train)
        #
        # print("Meilleur score de fitness atteint:", winner.fitness)
        #
        # generation_fitnesses = [c.fitness for c in stats.most_fit_genomes]
        # print("Scores de fitness par génération:", generation_fitnesses)
        #
        # # print(f"Meilleur génome : {winner}")
        # with open('best_model.pkl', 'wb') as f:
        #     pickle.dump(winner, f)
        # # sys.exit()

        stats = neat.StatisticsReporter()
        self.p.add_reporter(stats)

        executions = compteur.compter_executions()

        # Tu peux aussi ajouter un Checkpointer si tu veux
        # self.p.add_reporter(neat.Checkpointer(10))  # sauvegarde tous les 10 générations

        # Excel : fichier, workbook, worksheet
        fichier, wb, ws = exw.create("donnees_neat", f"entrainement_{executions}", "Génération", "Fitness")

        for generation in range(nb_loop_train):
            winner = self.p.run(self.eval_genomes, 1)  # run une génération

            # Fitness moyenne de la génération
            generation_fitnesses = [c.fitness for c in stats.most_fit_genomes]
            mean_fitness = sum(generation_fitnesses) / len(generation_fitnesses)

            print(f"Génération {generation}, fitness moyenne: {mean_fitness}, meilleure: {winner.fitness}")

            # Envoi dans Excel
            exw.ajouter_donnee(
                fichier, wb, ws,
                generation,
                mean_fitness,
                "Évolution de la fitness NEAT",
                "Génération",
                "Fitness"
            )

        # Sauvegarde régulière du meilleur modèle
        executions = compteur.compter_executions()
        with open(f'best_n_{executions}.pkl', 'wb') as f:
            pickle.dump(winner, f)

        print("Meilleur score de fitness atteint:", winner.fitness)

    def runBestNeat(self):
        with open('best_model_11.pkl', 'rb') as f:
            winner = pickle.load(f)

        # time.sleep(2)
        # winner_net = Neat()

        winner_net = neat.nn.FeedForwardNetwork.create(winner, self.config)

        score = snake.game_loop(snake.rect_width, snake.rect_height, snake.display, winner_net, None, 0, Neat)

        print(f"Score de {score}\n")

        print("Gènes des connexions :", winner.connections)
        print("Nombre de nœuds :", len(winner.nodes))
        print("Liste des nœuds :", winner.nodes.keys())

        for conn_key, conn_gene in winner.connections.items():
            source, target = conn_key
            status = "Activée" if conn_gene.enabled else "Désactivée"
            print(f"Connexion : {source} -> {target} | Poids : {conn_gene.weight:.3f} | {status}")

        visualize_neat_network(winner, self.config)


