import matplotlib.pyplot as plt
import pickle

task = "ToggleOn(CoffeeMachine)"

heuristic_data = pickle.load(open("complete_data/ToggleObjectOn_CoffeeMachine_floor1_heuristic_50.p","rb"))
heuristic_rate = float(len(heuristic_data["success"])) / (len(heuristic_data["success"]) + len(heuristic_data["failure"]))
print(heuristic_rate)

random_data = pickle.load(open("complete_data/ToggleObjectOn_CoffeeMachine_floor1_random_50.p","rb"))
random_rate = float(len(random_data["success"])) / (len(random_data["success"]) + len(random_data["failure"]))
print(random_rate)

plt.bar(["heuristic","random"], [heuristic_rate,random_rate])
plt.xlabel("Sampling method")
plt.ylabel("Success rate")
plt.title(task)
plt.show()

plt.savefig('random_vs_heuristic_successrate.png')
