print('hi')
x = input()
if x == '':
    print('empy')
split = x.split()
for i, val in enumerate(split):
    split[i] = int(val)

print(split)