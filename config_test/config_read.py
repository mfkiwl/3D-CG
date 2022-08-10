import yaml

with open("config_plane.yaml", 'r') as stream:
    dictionary = yaml.load(stream, Loader=yaml.loader.FullLoader)

for key, value in dictionary.items():
    print (key + " : " + str(value))
