from pysortd import SORTDClassifier
import pandas as pd

df = pd.read_csv('data/accuracy/monk2.csv',delimiter=" ", header=None)
X, y = df.iloc[:, 1:], df.iloc[:, 0]

model = SORTDClassifier("cost-complex-accuracy",max_depth = 3, cost_complexity=0.01, verbose=True,
                        use_rashomon_multiplier=True,rashomon_multiplier=0.1, max_num_trees=1000)
model.fit(X,y)
rashomon_set_size = model.rashomon_set_size
solutions = model.get_solution_list()

solution_values = []
for solution in solutions:
    values = [solution.objective / len(X) for d in range(solution.num_solutions)]
    solution_values.extend(values)
pass