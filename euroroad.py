# -*- coding: utf-8 -*-

import pandas as pd
import igraph as ig
from igraph import Graph, Layout
from matplotlib import pyplot as plt
import numpy as np
import random
import time as tm
import os,sys
import geopandas as gpd
from geopy.geocoders import Nominatim

#convert to layout coordinates for igraph
def convert_to_coords(lats,longs):
    coords = []
    for i in range(len(lats)): coords.append((longs[i],lats[i]))
    return coords

def plot_map(graph,layout,longmin,longmax,latmin,latmax,namepng,title,special_node = None):

    fig, gax = plt.subplots(figsize = (9.6,5.4)) #16:9

    #reading map
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world = world.set_index("iso_a3")

    #plotting only europe
    world.query("continent == 'Europe'").plot(ax = gax, edgecolor='black', color='green')
    world.query("continent == 'Asia'").plot(ax = gax, edgecolor='black', color='yellow')
    #world.query("continent == 'North America'").plot(ax = gax, edgecolor='black', color='blue')
    #world.query("continent == 'South America'").plot(ax = gax, edgecolor='black', color='peru')
    world.query("continent == 'Africa'").plot(ax = gax, edgecolor='black', color='orange')
    #world.query("continent == 'Oceania'").plot(ax = gax, edgecolor='black', color='violet')

    gax.set_xlabel('Longitude [°]')
    gax.set_ylabel('Latitude [°]')
    gax.set_xlim(longmin-1,longmax+1)
    gax.set_ylim(latmin-1,latmax+1)
    gax.set_title(title)

    color = "red"
    if (special_node):
        color = []
        for inode in range(g.vcount()):
            if (inode == special_node): color.append("fuchsia")
            else: color.append("red")

    ig.plot(graph, target=gax, layout = layout, vertex_size=0.5, edge_width=0.3, vertex_color = color, edge_color = "black")

    # manually handling axis
    xticks = np.linspace(longmin, longmax, 5)
    yticks = np.linspace(latmin, latmax, 5)
    gax.set_xticks(xticks)
    gax.set_yticks(yticks)

    # gax.spines['bottom'].set_patch_line
    gax.spines[:].set_linewidth(0.5)
    gax.spines[:].set_visible(True)

    plt.savefig(namepng,dpi = 400)

def plot_community(graph,communities,layout,longmin,longmax,latmin,latmax,namepng,title):

    fig, gax = plt.subplots(figsize = (9.6,5.4))

    #reading map
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world = world.set_index("iso_a3")

    #plotting only europe
    world.query("continent == 'Europe'").plot(ax = gax, edgecolor='black', color='green')
    world.query("continent == 'Asia'").plot(ax = gax, edgecolor='black', color='yellow')
    #world.query("continent == 'North America'").plot(ax = gax, edgecolor='black', color='blue')
    #world.query("continent == 'South America'").plot(ax = gax, edgecolor='black', color='peru')
    world.query("continent == 'Africa'").plot(ax = gax, edgecolor='black', color='orange')
    #world.query("continent == 'Oceania'").plot(ax = gax, edgecolor='black', color='violet')

    gax.set_xlabel('Longitude [°]')
    gax.set_ylabel('Latitude [°]')
    gax.set_xlim(longmin-1,longmax+1)
    gax.set_ylim(latmin-1,latmax+1)
    gax.set_title(title)

    num_communities = len(communities)
    palette = ig.ClusterColoringPalette(n = num_communities)
    for i, community in enumerate(communities):
        graph.vs[community]["color"] = i
        community_edges = graph.es.select(_within=community)
        community_edges["color"] = i

    ig.plot(communities, target=gax, layout = layout, vertex_size=0.5, edge_width=0.3, palette = palette)

    legend_handles = []
    for i in range(num_communities):
        handle = gax.scatter([],[],s=100, facecolor = palette.get(i), edgecolor="k", label=i)
        legend_handles.append(handle)

    '''
    gax.legend(
        handles = legend_handles,
        title = 'Component:',
        loc = "best",
        #bbox_to_anchor=(0, 1.0),
        bbox_transform = gax.transAxes,
    )
    '''

    # manually handling axis
    xticks = np.linspace(longmin, longmax, 5)
    yticks = np.linspace(latmin, latmax, 5)
    gax.set_xticks(xticks)
    gax.set_yticks(yticks)

    # gax.spines['bottom'].set_patch_line
    gax.spines[:].set_linewidth(0.5)
    gax.spines[:].set_visible(True)

    plt.savefig(namepng,dpi = 400)

#----------------------
#---------main---------
#----------------------

t1 = tm.time()
terminalout = open("euroroad/output.txt", 'w')
sys.stdout = terminalout

print("-------------------------")
print("EUROROAD NETWORK")

# load data into a graph
g = ig.Graph.Read_Ncol("euroroad/euroroad.txt", directed=False)

#reading cities
f = open("euroroad/euroroad_names.txt","r",encoding="utf8")
cities = f.readlines()

#reading latitudes and longitudes
lat_lon_df = pd.read_csv("euroroad/latlon.csv", sep = "\t")

#converting to lists
lats = lat_lon_df["LATITUDE"].values.tolist()
longs = lat_lon_df["LONGITUDE"].values.tolist()
lat_lon = lat_lon_df.values.tolist()

#converting latitude and long to grid coordinates
coords = convert_to_coords(lats,longs)
layout = Layout(coords)

#statistics of latitude and longitude
latmu, longmu = np.mean(lats), np.mean(longs)
latmin,latmax = min(lats), max(lats)
longmin, longmax = min(longs), max(longs)
print("Latitude Range: [%.2f,%.2f]" % (latmin, latmax))
print("Longitude Range: [%.2f,%.2f]" % (longmin, longmax))

# number of nodes
N = g.vcount()

# number of edges
m = g.ecount()

#density
D = g.density()  # if D<<1 we have a sparse graph (=> sparse adjacency matrix)

#order of nodes is mixed up...
realorder = []
for i in range(N): realorder.append(int(g.vs[i]['name'])-1)

#saving original variables...
lats_old, longs_old = lats,longs
layout_old, cities_old = layout,cities

#new ordered variables
layout = np.array(layout_old)[realorder].tolist()
lats = np.array(lats_old)[realorder].tolist()
longs = np.array(longs_old)[realorder].tolist()
cities = np.array(cities_old)[realorder].tolist()

print("\nCaratteristiche Rete Completa:")
print("Nodi = %i" % N)
print("Edges = %i" % m)
print("Densità = %f" % D)

# neighbors of a node i
list_of_neighbors = {f'{i}': g.neighbors(i) for i in range(N)}

#plotting map and igraph
plot_map(g,layout,longmin,longmax,latmin,latmax,"euroroad/figs/CompleteGraph.png","EuroRoad - Complete Network")

#connectivity
g.is_connected()

#Taking the biggest connected component and saving it in Gmax
components = g.connected_components() 
plot_community(g,components,layout,longmin,longmax,latmin,latmax,"euroroad/figs/CompleteGraph_Components.png","EuroRoad - Connected Components")

# list of list of components
list_of_components = [i for i in components]
comp = range(len(list_of_components))
list_of_sizes = [components.size(i) for i in comp]

plt.figure()
plt.plot(comp, list_of_sizes)
plt.title('Components Size')
plt.xlabel("Graph Component")
plt.ylabel("Number of Vertices")
plt.yscale("log")
plt.savefig('euroroad/figs/CompSize.png',dpi=280)

#taking the biggest connected
sorted_idx = np.argsort(list_of_sizes)
sorted_components = [list_of_components[i] for i in sorted_idx]
Gmax = components.subgraph(sorted_idx[-1])  # saving max component

# number of nodes
N = Gmax.vcount()

# number of edges
m = Gmax.ecount()

# DENSITY
D = Gmax.density()

# DIAMETER
diam = Gmax.diameter(directed=False)

#only connected nodes survive
realorder = []
for i in range(N): realorder.append(int(Gmax.vs[i]['name'])-1)
layout = np.array(layout_old)[realorder].tolist()
lats = np.array(lats_old)[realorder].tolist()
longs = np.array(longs_old)[realorder].tolist()
cities = np.array(cities_old)[realorder].tolist()

print("\nCaratteristiche Rete Connessa:")
print("Nodi = %i" % N)
print("Edges = %i" % m)
print("Densità = %f" % D)
print("Diametro = %f" % diam)

#plotting map and igraph
plot_map(Gmax,layout,longmin,longmax,latmin,latmax,"euroroad/figs/GraphMax.png","EuroRoad - Largest Connected Component")
plot_map(Gmax,layout,6.6272658, 18.7844746,35.2889616, 47.0921462,"euroroad/figs/GraphItaly.png","Italian (European) Road Network")

# SMALL WORLD
N1 = Gmax.vcount()
distances = list()
for i in range(0, N1):
    for j in range(i+1, N1):
        L = Gmax.get_shortest_paths(i, j)
        distance = len(L[0])-1
        distances.append(distance)
        
# Average shortest path between nodes:
Avg_shortest_path = int(np.mean(distances)+0.5)

print("Average Shortest Path = %.2f" % Avg_shortest_path)

# Average number of nodes within a distance l
minimum = int(np.min(distances)+0.5)
maximum = int(np.max(distances)+0.5)
x = list(range(minimum, maximum+1))

count = list()
for i in x:
    subdist = [d for d in distances if d <= i]
    count.append(len(subdist))
    
plt.figure()
plt.plot(x, count, color='green', linestyle='dashed', linewidth=2, marker='o',markerfacecolor='blue', markersize=6)
plt.yscale("log")
plt.ylabel("Events")
plt.xlabel("Distance")
plt.title('Distances Cumulative Distribution')
plt.savefig('euroroad/figs/DistancesCumulative.png',dpi=280)

nbinsdist=max(distances)-min(distances)+1
plt.figure()
plt.hist(distances,bins=nbinsdist,edgecolor='black')
plt.yscale("log")
plt.title('Distances Histogram')
plt.xlabel('Distance')
plt.ylabel('Events')
plt.savefig('euroroad/figs/DistancesHisto.png',dpi=280)

# BETWEENNESS
vertex_betweenness = Gmax.betweenness()
scaled_vertex_betweenness = ig.rescale(vertex_betweenness, clamp=True) #list of nodes betweenness

plt.figure()
plt.hist(scaled_vertex_betweenness,edgecolor='black')
plt.yscale("log")
plt.title('Vertex Betweenness Histogram')
plt.xlabel("Normalized Vertex Betweenness")
plt.ylabel("Events")
plt.savefig('euroroad/figs/BetwennessHisto.png',dpi=280)

# Cumulative distribution inversa
Range = list(range(int(min(vertex_betweenness)+0.5),int(max(vertex_betweenness)+0.5)+1))

count = list()
for i in Range:
    subbet = [d for d in vertex_betweenness if d >= i]
    count.append(len(subbet)/len(vertex_betweenness))

plt.figure()
plt.plot(Range, count, color='green', linestyle='solid', linewidth=2, marker='o', markerfacecolor='blue', markersize=5)
plt.yscale("log")
plt.xscale('log')
plt.xlabel("(Non-Normalized) Vertex Betweenness")
plt.ylabel("Events")
plt.title("Inverted Cumulative for Betweenness")
plt.savefig('euroroad/figs/BetwennessCumulative.png',dpi=280)

# max betweenness
node_maxbet = scaled_vertex_betweenness.index(max(scaled_vertex_betweenness))
print("Nodo con Max Betweenness: %s" % cities[node_maxbet])
lat,lon = lats[node_maxbet],longs[node_maxbet]
step_latlon = 2
plot_map(Gmax,layout,lon-step_latlon,lon+step_latlon,lat-step_latlon,lat+step_latlon,"euroroad/figs/MaxBetweenness.png","EuroRoad - Max Betweenness (%s)" % (cities[node_maxbet].strip()),node_maxbet)

# CLUSTERING
clustering_coef = Gmax.transitivity_local_undirected()

plt.figure()
plt.hist(clustering_coef,edgecolor='black')
plt.yscale("log")
plt.xlabel("Clustering Coefficient")
plt.ylabel("Events")
plt.title("Clustering Coefficient Histogram")
plt.savefig("euroroad/figs/ClusteringHisto.png",dpi=280)

#max clustering
node_maxclust = clustering_coef.index(max(clustering_coef))
print("Nodo con Max Clustering: %s" % cities[node_maxclust])
lat,lon = lats[node_maxclust],longs[node_maxclust]
step_latlon = 2
plot_map(Gmax,layout,lon-step_latlon,lon+step_latlon,lat-step_latlon,lat+step_latlon,"euroroad/figs/MaxClustering.png","Euroroad - Max Clustering (%s)" % (cities[node_maxclust].strip()),node_maxclust)

#DEGREE
H = Graph.degree(Gmax)

#max degree
node_maxdeg = H.index(max(H))
print("Nodo con Max Degree: %s" % cities[node_maxdeg])
lat,lon = lats[node_maxdeg],longs[node_maxdeg]
step_latlon = 10
plot_map(Gmax,layout,lon-step_latlon,lon+step_latlon,lat-step_latlon,lat+step_latlon,"euroroad/figs/MaxDegree.png","Euroroad - Max Degree (%s)" % (cities[node_maxdeg].strip()),node_maxdeg)


nbins = max(H)-min(H)+1
plt.figure()
plt.hist(H, bins = nbins, range = (min(H)-0.5,max(H)+0.5),edgecolor='black')
plt.yscale("log")
plt.xlabel("Vertex Degree")
plt.ylabel("Events")
plt.title("Degree Histogram")
plt.savefig("euroroad/figs/DegreeHisto.png",dpi=280)

# Degree distribution P(k) = N_k/N
minimum = min(H)
maximum = max(H)
Range = list(range(minimum, maximum+1))
P = [0 for i in Range]  # vector of P(K)

# Cumulative distribution
P_cumulative = np.cumsum(P)

# Cumulative distribution inversa
count = list()
for i in Range:
    subdeg = [d for d in H if d >= i]
    count.append(len(subdeg)/len(H))

plt.figure()
plt.plot(Range, count, color='green', linestyle='dashed', linewidth=2, marker='o', markerfacecolor='blue', markersize=5)
plt.yscale("log")
plt.xscale('log')
plt.xlabel("Vertex Degree")
plt.ylabel("Events")
plt.title("Inverted Cumulative for Betweenness")
plt.savefig("euroroad/figs/DegreeCumulative.png",dpi=280)

# Level of heterogeneity fluctuations
H_squared = [d**2 for d in H]
Level_of_het = np.mean(H_squared)/np.mean(H)

print("<H>: %f" % np.mean(H))
print("<H^2>: %f" % np.mean(H_squared))
print("Level of Heterogeneity: %f" % Level_of_het)

# Average degree of nearest neighbors
Avg_knn_degree = Graph.knn(Gmax)[0]

# Correlation Spectrum
Avg_knn_k = Graph.knn(Gmax)[1]  # it's nan if there is no vertex with such

# Assortative/Dissortative behaviour
plt.figure()
plt.plot(Range,Avg_knn_k, marker = 'o')
#plt.yscale("log")
#plt.xscale('log')
plt.xlabel("Vertex Degree")
plt.ylabel(r"$k_{nn}$(k)")
plt.title('Assortative/Dissortative Behaviour')
plt.savefig('euroroad/figs/AvgKNN',dpi=280)

#Average Clustering Coefficient with Same Degree
Avg_clust_k = list()

for i in Range:
    sum_degree = [c for ic,c in enumerate(clustering_coef) if H[ic] == i]
    if sum_degree: Avg_clust_k.append(np.mean(sum_degree))
    else: Avg_clust_k.append(np.nan)
 
plt.figure()
plt.plot(Range,Avg_clust_k, marker = 'o')  
plt.xlabel("Vertex Degree")
plt.ylabel("C(k)")
plt.title('Average Clustering Coefficient with Same Degree')
plt.savefig('euroroad/figs/AvgClust.png',dpi=280)

# COMMUNITY DETECTION
communities = Graph.community_multilevel(Gmax, return_levels = False)
# if return_levels = True, the communities at each level are returned in a list.
# if False, only the community structure with the best modularity is returned.
plot_community(Gmax,communities,layout,longmin,longmax,latmin,latmax,"euroroad/figs/CommunityMultiLevel.png","Euroroad - Communities (Multi-Level)")

#communities = Gmax.community_edge_betweenness()
#communities = communities.as_clustering()
#plot_community(Gmax,communities,layout,longmin,longmax,latmin,latmax,"euroroad/figs/CommunityEdgeBetweenness.png","Euroroad - Communities (Edge Betweenness)")

t2 = tm.time()-t1
print("\nAnalysis terminated in ",t2,"seconds.\n")

terminalout.close()