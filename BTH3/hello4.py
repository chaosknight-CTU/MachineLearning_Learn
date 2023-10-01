import pandas as pd
dulieu = pd.read_csv('winequality-white.csv', delimiter=';')



print("So thuoc tinh: \n", len(dulieu.columns)-1)

print("Cot nhan\n", dulieu.iloc[:, -1])
