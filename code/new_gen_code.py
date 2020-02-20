partials2 = []
for i in range(len(partials)):
    partials2.append(kda.generate_partial_diagrams(partials[i]))
partials3 = []
for i in range(len(partials2)):
    for j in range(len(partials2[i])):
        partials3.append(partials2[i][j])

partials3_edges = []
for i in range(len(partials3)):
    partials3_edges.append(partials3[i].edges)

edge_indices = np.unique(partials3_edges, return_index=True, axis=0)[-1]

partials4 = [partials3[i] for i in edge_indices]
# now we need to remove all of the closed loops

cycles = []
for i in range(len(unique_partials)):
    cycles.append([i, nx.is_directed_acyclic_graph(unique_partials[i])])

for i in range(len(partials3)):
    print(np.unique(np.sort(partials3[i].edges())))

def function(partials2):
    for i in range(len(partials2)):
        for j in range(len(partials2[i])):
            fig1 = plt.figure(figsize=(4, 3), tight_layout=True)
            fig1.add_subplot(111)
            partial = partials2[i][j]
            nx.draw_networkx_nodes(partial, pos, node_size=500, nodelist=[i for i in range(partials[0].number_of_nodes())], node_color='grey')
            nx.draw_networkx_edges(partial, pos, width=4, arrow_style='->', arrowsize=15)
            labels = {}
            for k in range(partials2[0][0].number_of_nodes()):
                labels[k] = r"${}$".format(k+1)
            nx.draw_networkx_labels(partial, pos, labels, font_size=16)
plt.show()
function(partials2)
