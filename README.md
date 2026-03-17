# 🐍 Snake AI Using NEAT (NeuroEvolution)

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Neat](https://img.shields.io/badge/Neat-0.92-red.svg)
![Numpy](https://img.shields.io/badge/Numpy-2.2.6-red.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.6.1-red.svg)
![OpenPyxl](https://img.shields.io/badge/OpenPyxl-3.1.5-red.svg)  

![License](https://img.shields.io/badge/license-MIT-green.svg)  
![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)  

<p align="center">
  <img src="Images/score13.gif" alt="Gif - AI with a score of 13">
</p>

## 📝 Project Description 
This project features an AI that learns to play [My Snake game](https://github.com/Thibault-GAREL/snake_game) autonomously using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. No hardcoded strategies — the agent improves over generations through genetic mutations and natural selection. 🧬🤖

---

## 🚀 Features
  🔄 No supervised learning – only evolution by fitness

  🧠 Networks evolve topologies and weights

  📊 Real-time simulation with visualization

  🏆 Tracks best fitness, average scores, and generation progress in Excel

---

## ⚙️ How it works

  🕹️ The AI controls a snake in my classic grid-based [Snake game](https://github.com/Thibault-GAREL/snake_game).

  🧬 It evolves over time using NEAT: networks mutate, reproduce, and get selected based on performance (fitness).

  👁️ Visual interface shows the best snake live as it learns.

  📈 An Excel is here to track the score or the loss.

## 🗺️ Schema
⏳ Training takes time – early generations play poorly but evolve quickly. I train it approximately 15h and the best score is more than 20 apples. It can also **adapt** to different area. Here is the best neural network :

![NN_snake](Images/network_graph.png)

🧪 You can adjust mutation rates, population size, and other parameters in the NEAT config file.

<details>
<summary>📸 See the neural network analysis</summary>


#### Distance to walls (8 inputs)

1. **distance_bord_n** — Distance to the wall to the North
2. **distance_bord_n_e** — Distance to the wall to the North-East 
3. **distance_bord_e** — Distance to the wall to the East 
4. **distance_bord_s_e** — Distance to the wall to the South-East 
5. **distance_bord_s** — Distance to the wall to the South 
6. **distance_bord_s_w** — Distance to the wall to the South-West 
7. **distance_bord_w** — Distance to the wall to the West 
8. **distance_bord_n_w** — Distance to the wall to the North-West

#### Distance to food (8 inputs)

9. **distance_food_n —** Distance to the food to the North 
10. **distance_food_n_e** — Distance to the food to the North-East 
11. **distance_food_e** — Distance to the food to the East 
12. **distance_food_s_e** — Distance to the food to the South-East 
13. **distance_food_s** — Distance to the food to the South 
14. **distance_food_s_w** — Distance to the food to the South-West 
15. **distance_food_w** — Distance to the food to the West 
16. **distance_food_n_w** — Distance to the food to the North-West


![NN_snake](Images/network_graph-without-unuseful-connection.png)

</details>

---

## 📂 Repository structure  
```bash
├── Images/                 # Images for the README
│
├── best_model_8.pkl        # Saved model checkpoint
├── best_model_11.pkl       # Saved model checkpoint - the best
│
├── compteur.py             # Counter script
├── compteur_executions.txt # Execution log for the counter
├── config.txt              # Neat configuration file
├── donnees_neat.xlsx       # Training data (Excel) - Graph for score
│
├── exw.py                  # Excel writer script
├── ia.py                   # Main AI logic
├── main.py                 # Project entry point
├── network_graph/          # Network graph visualization
│   └── network_graph.png
├── snake.py                # Snake game implementation
│
├── LICENSE                 # Project license
├── README.md               # Main documentation
```

---

## 💻 Run it on Your PC  
Clone the repository and install dependencies:  
```bash
git clone https://github.com/Thibault-GAREL/snake_game.git
cd snake_game

python -m venv .venv #if you don't have a virtual environnement
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

pip install neat-python numpy pygame openpyxl

python main.py
```
---

## 📖 Inspiration / Sources  
I code it without any help 😆 !

Code created by me 😎, Thibault GAREL - [Github](https://github.com/Thibault-GAREL)