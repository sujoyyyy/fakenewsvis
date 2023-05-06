from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
import uuid
import json
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import re
import os
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import random
import networkx as nx
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import Counter
import numpy as np
from scipy.integrate import odeint
from constants import alpha, beta, gamma, delta, a, c, d, f, g, c1, c2, c3, c4, c5, c6, b, epsilon

image = Image.open("TMGC+.png")
st.image(image,width=110, use_column_width=False)
# Render Streamlit page
st.title("Inductive Fake News Visualiser")
st.markdown("This mini-app visualises propagation of FakeNews using the SEIRMZ model. You can find the code on [GitHub](https://github.com/sujoyyyy/OSN-Graphs) and the author on [Twitter](https://twitter.com/chuphojasujoy).")

# Get user input for news headline
headline = st.text_area("Enter the news headline (up to 500 characters, Hindi/English):", height = 100, max_chars=500)
population_size = st.number_input("Enter the population size:", min_value=1, step=1, value=15)
num_skeptic_nodes = st.number_input("Number of skeptic nodes to add:", min_value=0, max_value=20, step=2)

# Define the file URL and output filename
file_url = "https://drive.google.com/uc?id=1IFQKH5TrhCGdqOXJ37ul5GIlqxkP9yv_&export=download"
filename = "bert_model.pt"

# Use wget to download the file
os.system(f"wget -O {filename} {file_url}")

# Load the saved model
checkpoint = torch.load('bert_model.pt')
# Create a new instance of the BERT model using the saved configuration and tokenizer
model = AutoModelForSequenceClassification.from_config(checkpoint['config'])
model.load_state_dict(checkpoint['state_dict'])
tokenizer = checkpoint['tokenizer']

#defining compartments
compartments = ['S', 'E', 'I', 'R', 'M', 'Z']

# Add your key and endpoint
key = "b7d9de81f743451686502350c1e39daf"
endpoint = "https://api.cognitive.microsofttranslator.com"

# location, also known as region.
# required if you're using a multi-service or regional (not global) resource. It can be found in the Azure portal on the Keys and Endpoint page.
location = "centralindia"

path = '/translate'
constructed_url = endpoint + path

params = {
    'api-version': '3.0',
    'from': 'hi',
    'to': ['en']
}

headers = {
    'Ocp-Apim-Subscription-Key': key,
    # location required if you're using a multi-service or regional (not global) resource.
    'Ocp-Apim-Subscription-Region': location,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}

#Function to create a Graph
def create_graph(N):
    G = nx.Graph()
    for i in range(N):
        G.add_node(i)

    # assign at least one node to each compartment
    compartment_nodes = {}
    for compartment in compartments:
        node = random.choice(list(G.nodes()))
        compartment_nodes[compartment] = node
        G.nodes[node]['compartment'] = compartment

    # assign remaining nodes to compartments randomly
    for node in G.nodes():
        if 'compartment' not in G.nodes[node]:
            compartment = random.choice(list(compartments))
            G.nodes[node]['compartment'] = compartment

    # add edges randomly between nodes
    for i in range(N):
        for j in range(i+1, N):
            if random.random() < 0.3:
                G.add_edge(i, j)

    # calculate edge weights using jaccard similarity index
    for u, v in G.edges():
        compartment1 = G.nodes[u]['compartment']
        compartment2 = G.nodes[v]['compartment']
        if compartment1 == compartment2:
            weight = 0.0
        else:
            neighbors1 = set(G.neighbors(u))
            neighbors2 = set(G.neighbors(v))
            weight = len(neighbors1.intersection(neighbors2)) / len(neighbors1.union(neighbors2))
        G[u][v]['weight'] = weight
    
    return G

#Function to Plot a Graph
def plot_graph(G):
    # Assign different colors to nodes based on their compartments
    node_colors = []
    for node in G.nodes:
        compartment = G.nodes[node]['compartment']
        if compartment == 'S':
            node_colors.append('blue')
        elif compartment == 'E':
            node_colors.append('green')
        elif compartment == 'I':
            node_colors.append('red')
        elif compartment == 'R':
            node_colors.append('orange')
        elif compartment == 'M':
            node_colors.append('purple')
        elif compartment == 'Z':
            node_colors.append('brown')
    
    # Plot the graph in shell layout
    pos = nx.shell_layout(G)
    nx.draw(G, pos=pos, node_color=node_colors, with_labels=True)
    #plt.show()
    st.pyplot(plt)

#Function to print Strength of Connection(Friendship)
def edge_weights_show(G):
    # print edge list with weights
    for u, v, w in G.edges(data=True):
        print(f"{u} -- {v} : {w['weight']}")


def adjacency_mat_plot(G):
    adj_mat = nx.to_numpy_matrix(G)
    df = pd.DataFrame(adj_mat, index=G.nodes(), columns=G.nodes())
    st.write("Adjacency Matrix:\n")
    # Convert the DataFrame to a formatted string
    table_str = tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False)
    # Display the table in Streamlit
    st.write(table_str)

def compartment_info_tab(G):
    compartment_info = {}
    for node in G.nodes():
        compartment = G.nodes[node]['compartment']
        if compartment not in compartment_info:
            compartment_info[compartment] = []
        compartment_info[compartment].append(node)

    compartment_lengths = [len(v) for v in compartment_info.values()]
    max_compartment_length = max(compartment_lengths)

    for compartment in compartment_info:
        while len(compartment_info[compartment]) < max_compartment_length:
            compartment_info[compartment].append('')
            
    df = pd.DataFrame(compartment_info)
    df.index.name = 'Compartment'
    st.write("Compartment Info:")
    # Convert the DataFrame to markdown
    markdown_str = df.to_markdown(index=False)
    # Display the markdown as an HTML table
    st.write(markdown_str, unsafe_allow_html=True)

def deriv_seirmz(y, t, N, alpha, beta, gamma, delta, a, c, d, f, g, b, c1, c2, c3, c4, c5, c6, epsilon):
    S, E, I, R, M, Z = y
    dSdt = -(alpha * S * E) - (alpha * I * S) + ((c1 - c2) * epsilon)
    dEdt = (alpha * S * E) + (alpha * I * S) - (a * E) - (b * delta * E) - (c * E) - (c3 * epsilon)
    dIdt = (a * E - beta * I) - (b * gamma * I) - (d * I) + (f * Z)  - (c4 * epsilon)
    dRdt = (beta * I) + (g * Z) - (c5 * epsilon)
    dMdt = (delta * b * E) + (gamma * b * I)
    dZdt = (c * E) + (d * I) - (f * Z) - (g * Z) - (c6 * epsilon)
    return dSdt, dEdt, dIdt, dRdt, dMdt, dZdt


def seirmz_sim(G):
    # Create a list of compartment labels for all nodes in the graph
    compartment_labels = [G.nodes[node]['compartment'] for node in G.nodes()]

    # Count the number of nodes in each compartment
    node_counts = Counter(compartment_labels)

    # Print the node counts for each compartment
    for compartment in compartments:
        count = node_counts[compartment]
        print(f"Number of nodes in compartment {compartment}: {count}")
    S0 = node_counts['S']
    E0 = node_counts['E']
    I0 = node_counts['I']
    R0 = node_counts['R']
    M0 = node_counts['M']
    Z0 = node_counts['Z']
    # A grid of time points (in days)
    t = np.linspace(0, 160, 160)
    # Initial conditions vector
    y0 = S0, E0, I0, R0, M0, Z0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv_seirmz, y0, t, args=(N, alpha, beta, gamma, delta, a, c, d, f, g, b, c1, c2, c3, c4, c5, c6, epsilon))
    S, E, I, R, M, Z = ret.T

    # Plot the data on four separate curves for S(t), I(t), R(t) and M(t)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#dddddd')
    ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, E, 'c', alpha=0.5, lw=2, label='Exposed')
    ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.plot(t, M, 'y', alpha=0.5, lw=2, label='Mortality')
    ax.plot(t, Z, 'k', alpha=0.5, lw=2, label='Skeptic')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Population')
    ax.set_ylim(0,N)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(True, which='major', axis='both', linestyle='-', linewidth=2, color='w')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    #plt.show()
    st.pyplot(plt)


# Function to translate text using the Azure Translator API
def translate_text(text):
    body = [{
        'text': text
    }]        
    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()
    return response[0]['translations'][0]['text']

def preprocess_text(text):
    # Remove all URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove all non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove all special symbols
    text = re.sub(r'[^\w\s]', '', text)
    
    # Convert all text to lowercase
    text = text.lower()
    
    # Tokenize the text into words
    tokens = word_tokenize(text)
    
    # Remove all stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Join the tokens back into a single string
    text = ' '.join(filtered_tokens)
    
    return text

import random
def add_skeptic_nodes(G, n):
    new_nodes = range(len(G.nodes()), len(G.nodes())+n)
    for i in new_nodes:
        G.add_node(i, compartment='Z')
        for j in range(len(G.nodes())):
            if j != i and random.random() < 0.3:
                G.add_edge(i, j)
    
    # calculate edge weights using jaccard similarity index
    for u, v in G.edges():
        compartment1 = G.nodes[u]['compartment']
        compartment2 = G.nodes[v]['compartment']
        if compartment1 == compartment2:
            weight = 0.0
        else:
            neighbors1 = set(G.neighbors(u))
            neighbors2 = set(G.neighbors(v))
            weight = len(neighbors1.intersection(neighbors2)) / len(neighbors1.union(neighbors2))
        G[u][v]['weight'] = weight
    return G

def add_skeptic_nodes_max(G, n):
    """
    Add n skeptic nodes to Graph G, and connect them to all other nodes in a way that the new nodes have the highest degree.
    """
    z_nodes = [node for node, data in G.nodes(data=True) if data['compartment'] == 'Z']
    for i in range(n):
        # Add the new node to the graph with a 'Z' compartment label
        G.add_node(len(G), compartment='Z')
        # Connect the new node to all other nodes
        for node in z_nodes:
            G.add_edge(node, len(G)-1)
        z_nodes.append(len(G)-1)
    for u, v in G.edges():
        compartment1 = G.nodes[u]['compartment']
        compartment2 = G.nodes[v]['compartment']
        if compartment1 == compartment2:
            weight = 0.0
        else:
            neighbors1 = set(G.neighbors(u))
            neighbors2 = set(G.neighbors(v))
            weight = len(neighbors1.intersection(neighbors2)) / len(neighbors1.union(neighbors2))
        G[u][v]['weight'] = weight
    return G


def add_nodes_with_highest_betweenness(G, n):
    # Add n nodes to the Z compartment
    z_nodes = []
    for i in range(n):
        node_id = G.number_of_nodes()
        G.add_node(node_id, compartment='Z')
        z_nodes.append(node_id)

    # Calculate betweenness centrality
    betweenness = nx.betweenness_centrality(G)

    # Connect new nodes to other nodes based on highest betweenness
    for z_node in z_nodes:
        max_betweenness = -1
        max_node = None
        for node in G.nodes():
            if G.nodes[node]['compartment'] != 'Z':
                if betweenness[node] > max_betweenness:
                    max_betweenness = betweenness[node]
                    max_node = node
        G.add_edge(z_node, max_node)
    for u, v in G.edges():
        compartment1 = G.nodes[u]['compartment']
        compartment2 = G.nodes[v]['compartment']
        if compartment1 == compartment2:
            weight = 0.0
        else:
            neighbors1 = set(G.neighbors(u))
            neighbors2 = set(G.neighbors(v))
            weight = len(neighbors1.intersection(neighbors2)) / len(neighbors1.union(neighbors2))
        G[u][v]['weight'] = weight
    return G

import importlib
import constants
importlib.reload(constants)

def calculate_updated_f(G):
    total_weight = 0
    num_edges = 0
    for u, v in G.edges():
        if G.nodes[u]['compartment'] == 'Z' and G.nodes[v]['compartment'] == 'I' or \
            G.nodes[u]['compartment'] == 'I' and G.nodes[v]['compartment'] == 'Z':
            total_weight += G[u][v]['weight']
            num_edges += 1
    if num_edges == 0:
        return 0
    else:
        return total_weight / num_edges

def calculate_updated_g(G):
    total_weight = 0
    num_edges = 0
    for u, v in G.edges():
        if G.nodes[u]['compartment'] == 'Z' and G.nodes[v]['compartment'] == 'R' or \
            G.nodes[u]['compartment'] == 'R' and G.nodes[v]['compartment'] == 'Z':
            total_weight += G[u][v]['weight']
            num_edges += 1
    if num_edges == 0:
        return 0
    else:
        return total_weight / num_edges


def new_seirmz_sim(G):
    # Create a list of compartment labels for all nodes in the graph
    compartment_labels = [G.nodes[node]['compartment'] for node in G.nodes()]

    # Count the number of nodes in each compartment
    node_counts = Counter(compartment_labels)

    # Print the node counts for each compartment
    for compartment in compartments:
        count = node_counts[compartment]
        print(f"Number of nodes in compartment {compartment}: {count}")
    S0 = node_counts['S']
    E0 = node_counts['E']
    I0 = node_counts['I']
    R0 = node_counts['R']
    M0 = node_counts['M']
    Z0 = node_counts['Z']
    t = np.linspace(0, 40, 5)
    # Initial conditions vector
    y0 = S0, E0, I0, R0, M0, Z0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv_seirmz, y0, t, args=(N, alpha, beta, gamma, delta, a, c, d_new, f_new, g_new, b, c1, c2, c3, c4, c5, c6, epsilon))
    S, E, I, R, M, Z = ret.T
    S = [round(x, 2) for x in S]
    E = [round(x, 2) for x in E]
    I = [round(x, 2) for x in I]
    R = [round(x, 2) for x in R]
    M = [round(x, 2) for x in M]
    Z = [round(x, 2) for x in Z]
    # Create a table of the integrated SEIRMZ values
    table_data = []
    for i in range(len(t)):
        row = [t[i], round(ret[i,0], 2), round(ret[i,1], 2), round(ret[i,2], 2), round(ret[i,3], 2), round(ret[i,4], 2), round(ret[i,5], 2)]
        table_data.append(row)

    headers = ['Time', 'S', 'E', 'I', 'R', 'M', 'Z']

    fig, ax = plt.subplots(figsize=(0.2,0.5))

    # hide axes
    ax.axis('off')
    ax.axis('tight')

    # create table
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.scale(0.5,1)

    # set font size for cells
    font_size = 8
    table.set_fontsize(font_size)

    # set font size for column headers
    cellDict = table.get_celld()
    for i in range(len(headers)):
        cellDict[(0,i)].set_text_props(weight='bold', fontsize=font_size)

    # set table properties
    table.auto_set_column_width(col=list(range(len(headers))))

    # display the table
    #plt.show()
    markdown_text = """
    They may not have much influence on the spread of misinformation. However, if they are added to influential nodes or if they form clusters with other sceptic nodes, they can have a significant impact.                                           
    Uniform rate of recovery over time. Randomly adding sceptic nodes to the network can have mixed effects, however, they do not necessarily have a huge impact on the rate of recovery.
    """
    st.pyplot(plt)
    st.markdown(markdown_text)
    # A grid of time points (in days)
    t = np.linspace(0, 160, 160)
    # Initial conditions vector
    y0 = S0, E0, I0, R0, M0, Z0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv_seirmz, y0, t, args=(G.number_of_nodes, alpha, beta, gamma, delta, a, c, d_new, f_new, g_new, b, c1, c2, c3, c4, c5, c6, epsilon))
    S, E, I, R, M, Z = ret.T

    # Plot the data on four separate curves for S(t), I(t), R(t) and M(t)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#dddddd')
    ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, E, 'c', alpha=0.5, lw=2, label='Exposed')
    ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.plot(t, M, 'y', alpha=0.5, lw=2, label='Mortality')
    ax.plot(t, Z, 'k', alpha=0.5, lw=2, label='Skeptic')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Population')
    ax.set_ylim(0,N)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(True, which='major', axis='both', linestyle='-', linewidth=2, color='w')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    #plt.show()
    st.pyplot(plt)


def new_seirmz_sim_max(G):
    # Create a list of compartment labels for all nodes in the graph
    compartment_labels = [G.nodes[node]['compartment'] for node in G.nodes()]

    # Count the number of nodes in each compartment
    node_counts = Counter(compartment_labels)

    # Print the node counts for each compartment
    for compartment in compartments:
        count = node_counts[compartment]
        print(f"Number of nodes in compartment {compartment}: {count}")
    S0 = node_counts['S']
    E0 = node_counts['E']
    I0 = node_counts['I']
    R0 = node_counts['R']
    M0 = node_counts['M']
    Z0 = node_counts['Z']
    t = np.linspace(0, 40, 5)
    # Initial conditions vector
    y0 = S0, E0, I0, R0, M0, Z0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv_seirmz, y0, t, args=(N, alpha, beta, gamma, delta, a, c, d_new, f_new, g_new, b, c1, c2, c3, c4, c5, c6, epsilon))
    S, E, I, R, M, Z = ret.T
    S = [round(x, 2) for x in S]
    E = [round(x, 2) for x in E]
    I = [round(x, 2) for x in I]
    R = [round(x, 2) for x in R]
    M = [round(x, 2) for x in M]
    Z = [round(x, 2) for x in Z]

    # Create a table of the integrated SEIRMZ values
    table_data = []
    for i in range(len(t)):
        row = [t[i], round(ret[i,0], 2), round(ret[i,1], 2), round(ret[i,2], 2), round(ret[i,3], 2), round(ret[i,4], 2), round(ret[i,5], 2)]
        table_data.append(row)

    headers = ['Time', 'S', 'E', 'I', 'R', 'M', 'Z']

    fig, ax = plt.subplots(figsize=(0.2,0.5))

    # hide axes
    ax.axis('off')
    ax.axis('tight')

    # create table
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.scale(0.5,1)

    # set font size for cells
    font_size = 8
    table.set_fontsize(font_size)

    # set font size for column headers
    cellDict = table.get_celld()
    for i in range(len(headers)):
        cellDict[(0,i)].set_text_props(weight='bold', fontsize=font_size)

    # set table properties
    table.auto_set_column_width(col=list(range(len(headers))))

    # display the table
    #plt.show()
    markdown_text = """
    It is essential to note that the infection of the system is not directly impacted by the addition of sceptic nodes as max degree nodes.
    Based on our analysis of the network structure and dynamics, we can infer that adding sceptic nodes as the maximum degree nodes in an OSN can increase recovery rates.
    """
    st.pyplot(plt)
    st.markdown(markdown_text)
    # A grid of time points (in days)
    t = np.linspace(0, 160, 160)
    # Initial conditions vector
    y0 = S0, E0, I0, R0, M0, Z0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv_seirmz, y0, t, args=(G.number_of_nodes, alpha, beta, gamma, delta, a, c, d_new, f_new, g_new, b, c1, c2, c3, c4, c5, c6, epsilon))
    S, E, I, R, M, Z = ret.T

    # Plot the data on four separate curves for S(t), I(t), R(t) and M(t)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#dddddd')
    ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, E, 'c', alpha=0.5, lw=2, label='Exposed')
    ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.plot(t, M, 'y', alpha=0.5, lw=2, label='Mortality')
    ax.plot(t, Z, 'k', alpha=0.5, lw=2, label='Skeptic')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Population')
    ax.set_ylim(0,N)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(True, which='major', axis='both', linestyle='-', linewidth=2, color='w')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    #plt.show()
    st.pyplot(plt)

def new_seirmz_sim_b(G):
    # Create a list of compartment labels for all nodes in the graph
    compartment_labels = [G.nodes[node]['compartment'] for node in G.nodes()]

    # Count the number of nodes in each compartment
    node_counts = Counter(compartment_labels)

    # Print the node counts for each compartment
    for compartment in compartments:
        count = node_counts[compartment]
        print(f"Number of nodes in compartment {compartment}: {count}")
    S0 = node_counts['S']
    E0 = node_counts['E']
    I0 = node_counts['I']
    R0 = node_counts['R']
    M0 = node_counts['M']
    Z0 = node_counts['Z']
    t = np.linspace(0, 40, 5)
    # Initial conditions vector
    y0 = S0, E0, I0, R0, M0, Z0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv_seirmz, y0, t, args=(N, alpha, beta, gamma, delta, a, c, d_new, f_new, g_new, b, c1, c2, c3, c4, c5, c6, epsilon))
    S, E, I, R, M, Z = ret.T
    S = [round(x, 2) for x in S]
    E = [round(x, 2) for x in E]
    I = [round(x, 2) for x in I]
    R = [round(x, 2) for x in R]
    M = [round(x, 2) for x in M]
    Z = [round(x, 2) for x in Z]

    # Create a table of the integrated SEIRMZ values
    table_data = []
    for i in range(len(t)):
        row = [t[i], round(ret[i,0], 2), round(ret[i,1], 2), round(ret[i,2], 2), round(ret[i,3], 2), round(ret[i,4], 2), round(ret[i,5], 2)]
        table_data.append(row)

    headers = ['Time', 'S', 'E', 'I', 'R', 'M', 'Z']

    fig, ax = plt.subplots(figsize=(0.2,0.5))

    # hide axes
    ax.axis('off')
    ax.axis('tight')

    # create table
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.scale(0.5,1)

    # set font size for cells
    font_size = 8
    table.set_fontsize(font_size)

    # set font size for column headers
    cellDict = table.get_celld()
    for i in range(len(headers)):
        cellDict[(0,i)].set_text_props(weight='bold', fontsize=font_size)

    # set table properties
    table.auto_set_column_width(col=list(range(len(headers))))

    # display the table
    #plt.show()
    markdown_text = """
    Betweenness centrality measures how often a node lies on the shortest path between two other nodes. Adding sceptic nodes to these nodes can disrupt the flow of information in the network since they are key players in the spread of information.
    By identifying the nodes with the highest betweenness centrality, one can prioritize these nodes for recovery efforts. Rebuilding these key nodes first can help restore the flow of information in the network more quickly, as they are important players in the spread of information. Additionally, analyzing changes in betweenness centrality over time can provide insights into the effectiveness of recovery efforts and help identify areas that may still require attention.
    """
    st.pyplot(plt)
    st.markdown(markdown_text)
    # A grid of time points (in days)
    t = np.linspace(0, 160, 160)
    # Initial conditions vector
    y0 = S0, E0, I0, R0, M0, Z0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv_seirmz, y0, t, args=(G.number_of_nodes, alpha, beta, gamma, delta, a, c, d_new, f_new, g_new, b, c1, c2, c3, c4, c5, c6, epsilon))
    S, E, I, R, M, Z = ret.T

    # Plot the data on four separate curves for S(t), I(t), R(t) and M(t)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#dddddd')
    ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, E, 'c', alpha=0.5, lw=2, label='Exposed')
    ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.plot(t, M, 'y', alpha=0.5, lw=2, label='Mortality')
    ax.plot(t, Z, 'k', alpha=0.5, lw=2, label='Skeptic')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Population')
    ax.set_ylim(0,N)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(True, which='major', axis='both', linestyle='-', linewidth=2, color='w')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    #plt.show()
    st.pyplot(plt)

import pytablewriter

if st.button("Check News Quality"):
  translated_text = translate_text(headline)
  translated_text = preprocess_text(translated_text)
  encoded_input = tokenizer(translated_text, padding=True, truncation=True, max_length=256, return_tensors='pt')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  input_ids = encoded_input['input_ids'].to(device)
  attention_mask = encoded_input['attention_mask'].to(device)
  with torch.no_grad():
      output = model(input_ids, attention_mask=attention_mask)
  logits = output.logits
  prob = torch.softmax(logits, dim=1)
  label_indices = prob.argmax(dim=1)
  logits = output.logits
  prob = torch.softmax(logits, dim=1)
  fake_prob = prob[:, 0].item()
  fake_prob = fake_prob *100
  fake_prob = round(fake_prob, 2)
  st.write("The input has " + str(fake_prob) + "% probability of being fake news.")
  if fake_prob <= 35:
    markdown_text = """
    The news headline is classified as **real** or true.

    One of the reasons why **true news does not propagate as much as false news** is the inherent nature of human **psychology**. People tend to be drawn towards sensational or provocative stories, regardless of their veracity, as they tend to generate more interest and attention. False news, especially those that are alarming or controversial, often receive a lot of attention and shares on social media, which in turn increases their visibility and reach. This creates a vicious cycle where false news gets more attention, and hence, propagates much faster than true news.

    Another factor that contributes to the propagation of false news is the role of **social media algorithms**. 
    * Social media platforms often prioritize content that generates high engagement and interactions, such as likes, comments, and shares. False news stories, especially those that are sensational or controversial, tend to generate more engagement and hence, get boosted by social media algorithms.
    * As a result, true news stories, even those that are important or impactful, often get buried under the flood of false news. Furthermore, the rise of fake news and misinformation has led to increased distrust in the media, with many people struggling to distinguish between reliable sources and unreliable ones. This creates an environment where false news can easily spread, as people are less likely to fact-check and verify the news they consume.

    In contrast, true news requires more effort and time to verify and confirm, which makes it less likely to propagate quickly.

    In conclusion, the propagation of true news is often **hindered by various psychological, technological, and societal factors**. However, it is crucial that we make a concerted effort to combat fake news and misinformation, and prioritize the dissemination of accurate and verified information, to ensure a well-informed and educated society.
    """
    st.markdown(markdown_text)
  else:
    markdown_text = """Upon classification of the input as a fake headline, a study of its propagation is recommended to examine its dissemination patterns."""
    st.markdown(markdown_text)
    N = population_size
    G = create_graph(N)
    image = Image.open("legend.jpeg")
    st.image(image, caption="Compartment Legends", width=200)
    st.write("We first plot a social network graph with the mentioned population size. Please note each node has been colour coded based on the compartment it belongs to.")
    plot_graph(G)
    markdown_text = """
    We also plot the SEIRMZ compartmental model of ths OSN at T=0, based on the population size. Please note that the population is growing at a rate of 1 individual per day.
    """
    st.markdown(markdown_text)
    seirmz_sim(G)
    st.write("As per the user input, we add " + str(num_skeptic_nodes)+" to the OSN in different positions.")
    markdown_text = """
    ### Adding skeptic nodes randomly-
    Randomly adding skeptic nodes to the network can have mixed effects. 
    * Depending on where the skeptic nodes are added, they may not have much influence on the spread of misinformation. 
    * However, if they are added to influential nodes or if they form clusters with other skeptic nodes, they can have a significant impact on the spread of misinformation.
    """
    st.markdown(markdown_text)
    newG = add_skeptic_nodes(G,num_skeptic_nodes)
    st.write("The resultant graph looks like- ")
    k = calculate_updated_f(newG)
    f_new = f * k
    if k == 0:
        d_new = 0
    else:
        d_new = d/k
    g_new = g * calculate_updated_g(newG)
    new_seirmz_sim(newG)
    markdown_text = """
    ### Adding skeptic nodes at max degree-
    The nodes with the highest degree are the ones with the most connections. Adding skeptic nodes to these nodes can reduce the spread of misinformation since they will be able to influence a larger number of nodes. 
    * However, if these nodes are already spreading misinformation, adding skeptic nodes to them may not have much effect. 
    * The effect of adding skeptic nodes to high-degree nodes can be more pronounced if the nodes are central to the network.
    """
    st.markdown(markdown_text)
    newmaxdegreeG = add_skeptic_nodes_max(G,num_skeptic_nodes)
    st.write("The resultant graph looks like- ")
    k = calculate_updated_f(newmaxdegreeG)
    f_new = f * k
    if k == 0:
        d_new = 0
    else:
        d_new = d/k
    g_new = g * calculate_updated_g(newmaxdegreeG)
    new_seirmz_sim_max(newmaxdegreeG)
    markdown_text = """
    ### Adding skeptic nodes at max closeness-
    Betweenness centrality measures how often a node lies on the shortest path between two other nodes. 
    * Adding skeptic nodes to these nodes can disrupt the flow of information in the network since they are key players in the spread of information. 
    * Skeptic nodes added to these nodes may become bottlenecks for information flow and help in the reduction of the spread of misinformation.
    """
    st.markdown(markdown_text)
    nbG = add_nodes_with_highest_betweenness(G,num_skeptic_nodes)
    st.write("The resultant graph looks like- ")
    k = calculate_updated_f(nbG)
    f_new = f * k
    if k == 0:
        d_new = 0
    else:
        d_new = d/k
    g_new = g * calculate_updated_g(nbG)
    new_seirmz_sim_b(nbG)
