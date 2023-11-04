import json
import pandas as pd

json_file = open("USAir97v2.json")
     
# returns JSON object as a dictionary
graph_data = json.load(json_file)

# retrieving nodes
nodes = graph_data['nodes']

latlon = []
cities = []
for n in nodes:
    cities.append(n['city'])
    latlon.append([n['latitude'],n['longitude']])

header = "LATITUDE LONGITUDE"
header = header.split(sep = " ")

df = pd.DataFrame(data = latlon, columns = header)
df.to_csv("latlon.csv", index = False, sep = " ", mode = "w", header = True)

df = pd.DataFrame(data = cities)
df.to_csv("usair_names.txt", index = False, mode = "w", header = False)