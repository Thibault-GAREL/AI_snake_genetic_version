import neat
import os
import sys
import pickle
import numpy as np

import main

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
            genome.fitness = main.game_loop(main.rect_width, main.rect_height, main.display, net, genome,
                                       j)  # Ici je pense que il faut juste return le score simplement après une game
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



    def runNeat(self):
        stats = neat.StatisticsReporter()
        self.p.add_reporter(stats)

        winner = self.p.run(self.eval_genomes, 100000)

        print("Meilleur score de fitness atteint:", winner.fitness)

        generation_fitnesses = [c.fitness for c in stats.most_fit_genomes]
        print("Scores de fitness par génération:", generation_fitnesses)

        # print(f"Meilleur génome : {winner}")
        # with open('best_model.pkl', 'wb') as f:
        #     pickle.dump(winner, f)
        sys.exit()


# import neat
# import os
# import sys
# import pickle
# import numpy as np
#
# # import main
#
# class Neat:
#     def __init__(self):
#         self.initNeat()
#     def initNeat(self):
#         local_dir = os.path.dirname(__file__)
#         config_path = os.path.join(local_dir, "config.txt")
#         self.config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
#                                          neat.DefaultStagnation, config_path)
#         self.p = neat.Population(self.config)
#         self.p.add_reporter(neat.StdOutReporter(True))
#         self.reporter = neat.StatisticsReporter()
#         self.p.add_reporter(self.reporter)
#
#         # self.controlplayer = False
#
#         self.best_fitnesses = []
#         self.mean_fitnesses = []
#
#         self.alpha = 0.0
#         self.omega = 8.0  # Durée d'une game en secondes
#
#         # self.displayCollisionNodes()
#
#         # self.taskMgr.stop()
#         # self.taskMgr.add(self.updateCamera, "Update Camera")
#         # self.taskMgr.add(self.updatePlayerNeat, "Update Player")
#         # self.taskMgr.add(self.updateCollisionsNeat, "Update Collisions")
#         # self.taskMgr.add(self.verifFinPartie, "Vérification fin de Game")
#
#     def get_action(self, net, state):
#         return np.argmax(net.activate(state))
#
#     def tab_state(distance_bord_n, distance_bord_n_e, distance_bord_e, distance_bord_s_e, distance_bord_s,
#                   distance_bord_s_w, distance_bord_w, distance_bord_n_w, distance_food_n, distance_food_n_e,
#                   distance_food_e, distance_food_s_e, distance_food_s, distance_food_s_w, distance_food_w,
#                   distance_food_n_w):
#         return [distance_bord_n, distance_bord_n_e, distance_bord_e, distance_bord_s_e, distance_bord_s, distance_bord_s_w,
#              distance_bord_w, distance_bord_n_w, distance_food_n, distance_food_n_e, distance_food_e, distance_food_s_e,
#              distance_food_s, distance_food_s_w, distance_food_w, distance_food_n_w]
#
#
#
#     def runNeat(self):
#         winner = self.p.run(main.eval_genomes, 10000)
#         # print(f"Meilleur génome : {winner}")
#         # with open('best_model.pkl', 'wb') as f:
#         #     pickle.dump(winner, f)
#         sys.exit()
#
