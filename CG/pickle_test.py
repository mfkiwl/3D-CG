import pickle

a = ['hi', 'hello']

with open('pickle_test', 'wb') as file:
    pickle.dump(a, file)

with open('pickle_test', 'rb') as file:
    b = pickle.load(file)

print(b)