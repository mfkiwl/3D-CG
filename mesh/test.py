def test():
    print(foo)

def main(t):

    global foo
    foo = t
    print(t)
    test()

if __name__ == '__main__':
    mesh = {'t': 3}
    main(mesh['t'])