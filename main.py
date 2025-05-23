import pygame
import sys
import time
import random
from dataclasses import dataclass

from graphviz import Digraph

@dataclass
class Snake:
    x: int
    y: int

class Manager_snake:
    def __init__(self):
        self.list_snake = []
        self.lenght = 0
        self.direction = "RIGHT"
        self.moved = True  # Ajout du drapeau moved

    def add_snake(self, added_snake):
        self.list_snake.append(added_snake)
        self.lenght += 1

    def draw_snake(self):
        for index, selected_snake in enumerate(self.list_snake, start=0):
            pygame.draw.rect(display, GREEN, (selected_snake.x, selected_snake.y, rect_width, rect_height))
            # draw_cherckerboard()
            if index == 0:
                if self.direction == "UP":
                    pygame.draw.rect(display, BLACK, (selected_snake.x + 10, selected_snake.y + 5, 5, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_width - 15, selected_snake.y + 5, 5, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, 5, rect_height))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_width - 5, selected_snake.y, 5, rect_height))
                if self.direction == "DOWN":
                    pygame.draw.rect(display, BLACK, (selected_snake.x + 10, selected_snake.y + rect_width - 10, 5, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_width - 15, selected_snake.y + rect_width - 10, 5, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, 5, rect_height))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_width - 5, selected_snake.y, 5, rect_height))
                if self.direction == "RIGHT":
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_height - 10, selected_snake.y + 10, 5, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_height - 10, selected_snake.y + rect_height - 15, 5, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, rect_width, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y + rect_width - 5, rect_height, 5))
                if self.direction == "LEFT":
                    pygame.draw.rect(display, BLACK, (selected_snake.x + 5, selected_snake.y + 10, 5, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + 5, selected_snake.y + rect_height - 15, 5, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, rect_width, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y + rect_width - 5, rect_height, 5))

            if index == self.lenght-1:
                if self.list_snake[index].y != self.list_snake[index-1].y:
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, 5, rect_height))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_width - 5, selected_snake.y, 5, rect_height))
                if self.list_snake[index].x != self.list_snake[index-1].x:
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, rect_width, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y + rect_width - 5, rect_height, 5))

            if index < self.lenght-1 and index !=0:
                if (self.list_snake[index-1].x<selected_snake.x and selected_snake.x<self.list_snake[index+1].x) or (self.list_snake[index+1].x<selected_snake.x and selected_snake.x<self.list_snake[index-1].x):
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, rect_width, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y + rect_height - 5, rect_width, 5))
                if (self.list_snake[index-1].y<selected_snake.y and selected_snake.y<self.list_snake[index+1].y) or (self.list_snake[index+1].y<selected_snake.y and selected_snake.y<self.list_snake[index-1].y):
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, 5, rect_height))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_height - 5, selected_snake.y, 5, rect_width))
                if (self.list_snake[index].y < self.list_snake[index+1].y and self.list_snake[index].x < self.list_snake[index-1].x) or (self.list_snake[index].y < self.list_snake[index-1].y and self.list_snake[index].x < self.list_snake[index+1].x):
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, 5, rect_height))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, rect_width, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_width - 5, selected_snake.y + rect_height - 5, 5, 5))
                if (self.list_snake[index].y > self.list_snake[index+1].y and self.list_snake[index].x < self.list_snake[index-1].x) or (self.list_snake[index].y > self.list_snake[index-1].y and self.list_snake[index].x < self.list_snake[index+1].x):
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, 5, rect_height))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y + rect_height - 5, rect_width, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_width - 5, selected_snake.y, 5, 5))
                if (self.list_snake[index].y > self.list_snake[index+1].y and self.list_snake[index].x > self.list_snake[index-1].x) or (self.list_snake[index].y > self.list_snake[index-1].y and self.list_snake[index].x > self.list_snake[index+1].x):
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y + rect_width - 5, rect_width, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_width - 5, selected_snake.y, 5, rect_height))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, 5, 5))
                if (self.list_snake[index].y < self.list_snake[index+1].y and self.list_snake[index].x > self.list_snake[index-1].x) or (self.list_snake[index].y < self.list_snake[index-1].y and self.list_snake[index].x > self.list_snake[index+1].x):
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y, rect_width, 5))
                    pygame.draw.rect(display, BLACK, (selected_snake.x + rect_width - 5, selected_snake.y, 5, rect_height))
                    pygame.draw.rect(display, BLACK, (selected_snake.x, selected_snake.y + rect_height - 5, 5, 5))

    def move(self):
        head_x = self.list_snake[0].x
        head_y = self.list_snake[0].y

        if self.direction == "UP":
            head_y -= 50
        if self.direction == "DOWN":
            head_y += 50
        if self.direction == "RIGHT":
            head_x += 50
        if self.direction == "LEFT":
            head_x -= 50

        new_head = Snake(head_x, head_y)

        if not ((0 <= new_head.x < width) and (0 <= new_head.y < height)):
            return False

        if new_head in self.list_snake:
            return False

        self.list_snake.insert(0, new_head)
        self.list_snake.pop(-1)
        self.moved = True
        return True

    def print_snake(self):
        for index, selected_snake in enumerate(self.list_snake, start=0):
            print(f"snake n°{index} = {selected_snake}")

@dataclass
class food:
    x: int
    y: int

def generated_food (my_snake):
    list_position = []
    for x in range (0, int(width/rect_width)):
        for y in range(0, int(height/rect_height)):
            list_position.append((x * rect_width, y * rect_height))

    if not list_position:
        print("C'est gagné!")
    for co_snake in my_snake.list_snake:
        if (co_snake.x, co_snake.y) in list_position:
            list_position.remove((co_snake.x, co_snake.y))
    random_food = random.choice(list_position)
    return food(random_food[0], random_food[1])


def distance_bord_west (my_snake):
    distance_mini = my_snake.list_snake[0].x
    for snake_selected in my_snake.list_snake:
        if snake_selected.y == my_snake.list_snake[0].y and snake_selected.x < my_snake.list_snake[0].x:
            distance_temp = my_snake.list_snake[0].x - snake_selected.x
            if distance_temp < distance_mini:
                distance_mini = distance_temp
    return distance_mini

def distance_bord_est (my_snake):
    distance_mini = width - 50 - my_snake.list_snake[0].x
    for snake_selected in my_snake.list_snake:
        if snake_selected.y == my_snake.list_snake[0].y and snake_selected.x > my_snake.list_snake[0].x:
            distance_temp = snake_selected.x - my_snake.list_snake[0].x
            if distance_temp < distance_mini:
                distance_mini = distance_temp
    return distance_mini

def distance_bord_north (my_snake):
    distance_mini = my_snake.list_snake[0].y
    for snake_selected in my_snake.list_snake:
        if snake_selected.x == my_snake.list_snake[0].x and snake_selected.y < my_snake.list_snake[0].y:
            distance_temp = my_snake.list_snake[0].y - snake_selected.y
            if distance_temp < distance_mini:
                distance_mini = distance_temp
    return distance_mini

def distance_bord_south (my_snake):
    distance_mini = height - 50 - my_snake.list_snake[0].y
    for snake_selected in my_snake.list_snake:
        if snake_selected.x == my_snake.list_snake[0].x and snake_selected.y > my_snake.list_snake[0].y:
            distance_temp = snake_selected.y - my_snake.list_snake[0].y
            if distance_temp < distance_mini:
                distance_mini = distance_temp
    return distance_mini

def distance_bord_north_est (my_snake):
    distance_mini = ((my_snake.list_snake[0].y)**2 + (my_snake.list_snake[0].y)**2)**0.5 #distance north
    if distance_mini > ((width - 50 - my_snake.list_snake[0].x)**2 + (width - 50 - my_snake.list_snake[0].x)**2)**0.5:
        distance_mini = ((width - 50 - my_snake.list_snake[0].x)**2 + (width - 50 - my_snake.list_snake[0].x)**2)**0.5 #distance est
    for snake_selected in my_snake.list_snake:
        if my_snake.list_snake[0].x != snake_selected.x and my_snake.list_snake[0].y != snake_selected.y:
            pente = (my_snake.list_snake[0].x - snake_selected.x) / (my_snake.list_snake[0].y - snake_selected.y)
            if pente == -1 and my_snake.list_snake[0].x < snake_selected.x:
                distance = ((snake_selected.x - my_snake.list_snake[0].x)**2 + (snake_selected.y - my_snake.list_snake[0].y)**2)**0.5
                if distance < distance_mini:
                    distance_mini = distance
    return distance_mini

def distance_bord_south_est (my_snake):
    distance_mini = ((height - 50 - my_snake.list_snake[0].y)**2 + (height - 50 - my_snake.list_snake[0].y)**2)**0.5 #distance south
    if distance_mini > ((width - 50 - my_snake.list_snake[0].x)**2 + (width - 50 - my_snake.list_snake[0].x)**2)**0.5:
        distance_mini = ((width - 50 - my_snake.list_snake[0].x)**2 + (width - 50 - my_snake.list_snake[0].x)**2)**0.5 #distance est
    for snake_selected in my_snake.list_snake:
        if my_snake.list_snake[0].x != snake_selected.x and my_snake.list_snake[0].y != snake_selected.y:
            pente = (my_snake.list_snake[0].x - snake_selected.x) / (my_snake.list_snake[0].y - snake_selected.y)
            if pente == 1 and my_snake.list_snake[0].x < snake_selected.x:
                distance = ((snake_selected.x - my_snake.list_snake[0].x)**2 + (snake_selected.y - my_snake.list_snake[0].y)**2)**0.5
                if distance < distance_mini:
                    distance_mini = distance
    return distance_mini

def distance_bord_south_west (my_snake):
    distance_mini = ((height - 50 - my_snake.list_snake[0].y)**2 + (height - 50 - my_snake.list_snake[0].y)**2)**0.5 #distance south
    if distance_mini > ((my_snake.list_snake[0].x)**2 + (my_snake.list_snake[0].x)**2)**0.5:
        distance_mini = ((my_snake.list_snake[0].x)**2 + (my_snake.list_snake[0].x)**2)**0.5 #distance west
    for snake_selected in my_snake.list_snake:
        if my_snake.list_snake[0].x != snake_selected.x and my_snake.list_snake[0].y != snake_selected.y:
            pente = (my_snake.list_snake[0].x - snake_selected.x) / (my_snake.list_snake[0].y - snake_selected.y)
            if pente == -1 and snake_selected.x < my_snake.list_snake[0].x:
                distance = ((snake_selected.x - my_snake.list_snake[0].x)**2 + (snake_selected.y - my_snake.list_snake[0].y)**2)**0.5
                if distance < distance_mini:
                    distance_mini = distance
    return distance_mini

def distance_bord_north_west (my_snake):
    distance_mini = ((my_snake.list_snake[0].y)**2 + (my_snake.list_snake[0].y)**2)**0.5 #distance north
    if distance_mini > ((my_snake.list_snake[0].x)**2 + (my_snake.list_snake[0].x)**2)**0.5:
        distance_mini = ((my_snake.list_snake[0].x)**2 + (my_snake.list_snake[0].x)**2)**0.5 #distance west
    for snake_selected in my_snake.list_snake:
        if my_snake.list_snake[0].x != snake_selected.x and my_snake.list_snake[0].y != snake_selected.y:
            pente = (my_snake.list_snake[0].x - snake_selected.x) / (my_snake.list_snake[0].y - snake_selected.y)
            if pente == 1 and snake_selected.x < my_snake.list_snake[0].x:
                distance = ((snake_selected.x - my_snake.list_snake[0].x)**2 + (snake_selected.y - my_snake.list_snake[0].y)**2)**0.5
                if distance < distance_mini:
                    distance_mini = distance
    return distance_mini

def distance_food_north (my_snake, food):
    distance = 0
    if food.y < my_snake.list_snake[0].y and food.x == my_snake.list_snake[0].x:
        distance = my_snake.list_snake[0].y - food.y
    return distance

def distance_food_est (my_snake, food):
    distance = 0
    if my_snake.list_snake[0].x < food.x and food.y == my_snake.list_snake[0].y:
        distance = food.x - my_snake.list_snake[0].x
    return distance

def distance_food_south (my_snake, food):
    distance = 0
    if my_snake.list_snake[0].y < food.y and food.x == my_snake.list_snake[0].x:
        distance = food.y - my_snake.list_snake[0].y
    return distance

def distance_food_west (my_snake, food):
    distance = 0
    if food.x < my_snake.list_snake[0].x and food.y == my_snake.list_snake[0].y:
        distance = my_snake.list_snake[0].x - food.x
    return distance

def distance_food_north_est (my_snake, food):
    distance = 0
    if my_snake.list_snake[0].x != food.x and my_snake.list_snake[0].y != food.y:
        pente = (my_snake.list_snake[0].x - food.x) / (my_snake.list_snake[0].y - food.y)
        if pente == -1 and my_snake.list_snake[0].x < food.x:
            distance = ((food.x - my_snake.list_snake[0].x)**2 + (food.y - my_snake.list_snake[0].y)**2)**0.5
    return distance

def distance_food_south_est (my_snake, food):
    distance = 0
    if my_snake.list_snake[0].x != food.x and my_snake.list_snake[0].y != food.y:
        pente = (my_snake.list_snake[0].x - food.x) / (my_snake.list_snake[0].y - food.y)
        if pente == 1 and my_snake.list_snake[0].x < food.x:
            distance = ((food.x - my_snake.list_snake[0].x) ** 2 + (food.y - my_snake.list_snake[0].y) ** 2) ** 0.5
    return distance

def distance_food_south_west (my_snake, food):
    distance = 0
    if my_snake.list_snake[0].x != food.x and my_snake.list_snake[0].y != food.y:
        pente = (my_snake.list_snake[0].x - food.x) / (my_snake.list_snake[0].y - food.y)
        if pente == -1 and food.x < my_snake.list_snake[0].x:
            distance = ((my_snake.list_snake[0].x - food.x)**2 + (my_snake.list_snake[0].y - food.y)**2)**0.5
    return distance

def distance_food_north_west (my_snake, food):
    distance = 0
    if my_snake.list_snake[0].x != food.x and my_snake.list_snake[0].y != food.y:
        pente = (my_snake.list_snake[0].x - food.x) / (my_snake.list_snake[0].y - food.y)
        if pente == 1 and food.x < my_snake.list_snake[0].x:
            distance = ((my_snake.list_snake[0].x - food.x) ** 2 + (my_snake.list_snake[0].y - food.y) ** 2) ** 0.5
    return distance

def print_display(msg, color, position):
    txt = fonttype.render(msg, True, color)
    rect_text = txt.get_rect(**position)
    display.blit(txt, rect_text)

def draw_cherckerboard ():
    for x in range (0, int(width/rect_width)):
        for y in range(0, int(height/rect_height)):
            pygame.draw.line(display, BLACK_BRIGHTER, (x * rect_width, y * rect_height), (x * rect_width + rect_width, y * rect_height), 2)
            pygame.draw.line(display, BLACK_BRIGHTER, (x * rect_width, y * rect_height), (x * rect_width, y * rect_height + rect_height), 2)


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

def game_loop(rect_width, rect_height, display, net, genome, i):
    score = 0

    my_snake = Manager_snake()
    # first_snake = Snake(2 * 50, 3 * 50)  # ancienement : 5 * 50, 5 * 50
    first_snake = Snake(random.randint(0,7) * 50, random.randint(0,5) * 50)
    nb_direction = random.randint(0,3)
    if nb_direction == 0:
        my_snake.direction = "UP"
    if nb_direction == 0:
        my_snake.direction = "RIGHT"
    if nb_direction == 0:
        my_snake.direction = "DOWN"
    if nb_direction == 0:
        my_snake.direction = "LEFT"

    my_snake.add_snake(first_snake)

    # food_actuel = food(9 * 50, 5 * 50)
    food_actuel = generated_food(my_snake)

    running = True
    iteration = 0

    while running and iteration < 50:
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         running = False
        #     elif event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_ESCAPE:
        #             running = False
        #         if my_snake.moved:
        #             if event.key == pygame.K_UP and my_snake.direction != "DOWN":
        #                 my_snake.direction = "UP"
        #                 my_snake.moved = False  # Réinitialise moved
        #             if event.key == pygame.K_DOWN and my_snake.direction != "UP":
        #                 my_snake.direction = "DOWN"
        #                 my_snake.moved = False
        #             if event.key == pygame.K_RIGHT and my_snake.direction != "LEFT":
        #                 my_snake.direction = "RIGHT"
        #                 my_snake.moved = False
        #             if event.key == pygame.K_LEFT and my_snake.direction != "RIGHT":
        #                 my_snake.direction = "LEFT"
        #                 my_snake.moved = False

        #
        # display.fill(BLACK)
        #
        state = Neat.tab_state(distance_bord_north(my_snake), distance_bord_north_est(my_snake), distance_bord_est(my_snake), distance_bord_south_est(my_snake), distance_bord_south(my_snake),
                  distance_bord_south_west(my_snake), distance_bord_west(my_snake), distance_bord_north_west(my_snake), distance_food_north(my_snake, food_actuel), distance_food_north_est(my_snake, food_actuel),
                  distance_food_est(my_snake, food_actuel), distance_food_south_est(my_snake, food_actuel), distance_food_south(my_snake, food_actuel), distance_food_south_west(my_snake, food_actuel), distance_food_west(my_snake, food_actuel),
                  distance_food_north_west(my_snake, food_actuel))

        action = Neat.get_action(net, state)
        # print(f"Action : {action}")
        if action == 0 and my_snake.direction != "DOWN":  # î : 1 / -> : 2 / v : 3 / <- : 4
            my_snake.direction = "UP"
            my_snake.moved = False  # Réinitialise moved
        if action == 2 and my_snake.direction != "UP":
            my_snake.direction = "DOWN"
            my_snake.moved = False
        if action == 1 and my_snake.direction != "LEFT":
            my_snake.direction = "RIGHT"
            my_snake.moved = False
        if action == 3 and my_snake.direction != "RIGHT":
            my_snake.direction = "LEFT"
            my_snake.moved = False

        if my_snake.list_snake[0].x == food_actuel.x and my_snake.list_snake[0].y == food_actuel.y:
            snake_to_add = Snake(my_snake.list_snake[-1].x, my_snake.list_snake[-1].y)
            my_snake.add_snake(snake_to_add)
            food_actuel = generated_food(my_snake)
            score += 1

        if my_snake.move() == False:
            running = False



        # pygame.draw.rect(display, RED, (food_actuel.x, food_actuel.y, rect_width, rect_height))

        # my_snake.draw_snake()
        # my_snake.print_snake()



        # draw_cherckerboard()

        # print_display(f"Score : {score} / Itérations : {iteration} / Générations : {my_neat.p.generation} / Génome : {genome.key}", WHITE, {'topleft': (10, 10)})
        # print(f"lenght : {my_snake.lenght}")


        # pygame.display.update()
        # clock.tick(50)
        iteration += 1
    return score
    # pygame.quit()
    # sys.exit()

def game_loop_show(rect_width, rect_height, display, net, genome, i):
    score = 0

    my_snake = Manager_snake()
    first_snake = Snake(2 * 50, 3 * 50)  # ancienement : 5 * 50, 5 * 50
    # first_snake = Snake(random.randint(0,7) * 50, random.randint(0,5) * 50)
    # nb_direction = random.randint(0,3)
    # if nb_direction == 0:
    #     my_snake.direction = "UP"
    # if nb_direction == 0:
    #     my_snake.direction = "RIGHT"
    # if nb_direction == 0:
    #     my_snake.direction = "DOWN"
    # if nb_direction == 0:
    #     my_snake.direction = "LEFT"


    my_snake.add_snake(first_snake)

    food_actuel = food(5 * 50, 3 * 50)
    # food_actuel = generated_food(my_snake)

    running = True
    iteration = 0

    while running: #and iteration < 50
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if my_snake.moved:
                    if event.key == pygame.K_UP and my_snake.direction != "DOWN":
                        my_snake.direction = "UP"
                        my_snake.moved = False  # Réinitialise moved
                    if event.key == pygame.K_DOWN and my_snake.direction != "UP":
                        my_snake.direction = "DOWN"
                        my_snake.moved = False
                    if event.key == pygame.K_RIGHT and my_snake.direction != "LEFT":
                        my_snake.direction = "RIGHT"
                        my_snake.moved = False
                    if event.key == pygame.K_LEFT and my_snake.direction != "RIGHT":
                        my_snake.direction = "LEFT"
                        my_snake.moved = False


        display.fill(BLACK)

        state = Neat.tab_state(distance_bord_north(my_snake), distance_bord_north_est(my_snake), distance_bord_est(my_snake), distance_bord_south_est(my_snake), distance_bord_south(my_snake),
                  distance_bord_south_west(my_snake), distance_bord_west(my_snake), distance_bord_north_west(my_snake), distance_food_north(my_snake, food_actuel), distance_food_north_est(my_snake, food_actuel),
                  distance_food_est(my_snake, food_actuel), distance_food_south_est(my_snake, food_actuel), distance_food_south(my_snake, food_actuel), distance_food_south_west(my_snake, food_actuel), distance_food_west(my_snake, food_actuel),
                  distance_food_north_west(my_snake, food_actuel))

        action = Neat.get_action(net, state)
        # print(f"Action : {action}")
        if action == 0 and my_snake.direction != "DOWN":  # î : 1 / -> : 2 / v : 3 / <- : 4
            my_snake.direction = "UP"
            my_snake.moved = False  # Réinitialise moved
        if action == 2 and my_snake.direction != "UP":
            my_snake.direction = "DOWN"
            my_snake.moved = False
        if action == 1 and my_snake.direction != "LEFT":
            my_snake.direction = "RIGHT"
            my_snake.moved = False
        if action == 3 and my_snake.direction != "RIGHT":
            my_snake.direction = "LEFT"
            my_snake.moved = False

        if my_snake.list_snake[0].x == food_actuel.x and my_snake.list_snake[0].y == food_actuel.y:
            snake_to_add = Snake(my_snake.list_snake[-1].x, my_snake.list_snake[-1].y)
            my_snake.add_snake(snake_to_add)
            food_actuel = generated_food(my_snake)
            score += 1

        if my_snake.move() == False:
            running = False



        pygame.draw.rect(display, RED, (food_actuel.x, food_actuel.y, rect_width, rect_height))

        my_snake.draw_snake()
        # my_snake.print_snake()



        # draw_cherckerboard()

        # print_display(f"Score : {score} / Itérations : {iteration} / Générations : {my_neat.p.generation} / Génome : {genome.key}", WHITE, {'topleft': (10, 10)})
        # print(f"lenght : {my_snake.lenght}")


        pygame.display.update()
        clock.tick(5)
        iteration += 1
    return score
    # pygame.quit()
    # sys.exit()

import neat
import os
import sys
import pickle
import numpy as np

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
            genome.fitness = game_loop(rect_width, rect_height, display, net, genome,
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
        with open('best_model.pkl', 'wb') as f:
            pickle.dump(winner, f)
        # sys.exit()

    def runBestNeat(self):
        with open('best_model_8.pkl', 'rb') as f:
            winner = pickle.load(f)

        time.sleep(2)
        # winner_net = Neat()

        winner_net = neat.nn.FeedForwardNetwork.create(winner, self.config)

        score = game_loop_show(rect_width, rect_height, display, winner_net, None, 0)

        print(f"Score de {score}\n")

        print("Gènes des connexions :", winner.connections)
        print("Nombre de nœuds :", len(winner.nodes))
        print("Liste des nœuds :", winner.nodes.keys())

        for conn_key, conn_gene in winner.connections.items():
            source, target = conn_key
            status = "Activée" if conn_gene.enabled else "Désactivée"
            print(f"Connexion : {source} -> {target} | Poids : {conn_gene.weight:.3f} | {status}")

        visualize_neat_network(winner, self.config)

pygame.init()

width = 400
height = 300

rect_width = 50
rect_height = 50

# display = None
# clock = None
# fonttype = None


display = pygame.display.set_mode((width, height))
pygame.display.set_caption("Jeu snake")

clock = pygame.time.Clock()

fonttype = pygame.font.SysFont(None, 20)

BLACK = (40, 40, 60)
BLACK_BRIGHTER = (60, 60, 80)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 244)

my_neat = Neat()
# print(f"my neat.p : {my_neat.p}")
# my_neat.runNeat()

print("Run best net")
my_neat.runBestNeat()
