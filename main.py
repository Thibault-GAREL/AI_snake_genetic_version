import ia

import compteur

executions = compteur.compter_executions()
print(f"Exécution n°{executions}")


my_neat = ia.Neat()
# print(f"my neat.p : {my_neat.p}")
# my_neat.runNeat(ia.nb_loop_train)


print("Run best net")
my_neat.runBestNeat()

