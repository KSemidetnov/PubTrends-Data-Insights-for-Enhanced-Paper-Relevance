from flask import Flask, render_template, jsonify, render_template_string, request
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import requests
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import DBSCAN
import networkx as nx
import plotly.graph_objects as go
import logging
import plotly.offline as pyo

app = Flask(__name__)

# Главная страница с кнопкой
@app.route('/')
def index():
    return render_template_string('''
        <html>
            <head>
                <title>Главная</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: #f5f5f5;
                    }
                    h1 {
                        color: #333;
                        text-align: center;
                        margin-bottom: 20px;
                    }
                    button {
                        background-color: #4682B4;
                        color: white;
                        border: none;
                        padding: 10px 15px;
                        border-radius: 5px;
                        cursor: pointer;
                        font-weight: bold;
                        transition: background-color 0.3s;
                    }
                    button:hover {
                        background-color: #36648B;
                    }
                    input[type="text"] {
                        padding: 10px;
                        width: 70%;
                        border: 1px solid #ccc;
                        border-radius: 5px;
                        margin-right: 10px;
                    }
                    form {
                        display: flex;
                        align-items: center;
                    }
                    #content {
                        transition: opacity 0.3s;
                    }
                    #loading {
                        display: none;
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background-color: rgba(255, 255, 255, 0.8);
                        z-index: 1000;
                        justify-content: center;
                        align-items: center;
                    }
                    .spinner {
                        width: 100px;
                        height: 100px;
                        animation: spin 2s linear infinite;
                    }
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                </style>
            </head>
        <body>
            <div id="loading">
                <svg class="spinner" viewBox="0 0 50 50">
                    <circle cx="25" cy="25" r="20" fill="none" stroke="#4682B4" stroke-width="5" stroke-linecap="round">
                        <animate attributeName="stroke-dashoffset" dur="1.5s" repeatCount="indefinite" from="0" to="502"></animate>
                        <animate attributeName="stroke-dasharray" dur="1.5s" repeatCount="indefinite" values="150.6 100.4;1 250;150.6 100.4"></animate>
                    </circle>
                </svg>
            </div>
            
            <div id="content">
                <h1>PMID visualisation</h1>
                <br><br>
                <form action="/result" method="post" onsubmit="showLoading()">
                    <input type="text" name="user_input" placeholder="PMIDs in format: id1, id2, id3">
                    <button type="submit">Visualise</button>
                </form>
                <br><br>
                <a href="/hello" onclick="showLoading(event)">
                    <button>Use default data</button>
                </a>
            </div>

            <script>
                function showLoading(event) {
                    // If this is the default data link, prevent default and handle manually
                    if (event) {
                        event.preventDefault();
                    }
                    
                    // Hide the content
                    document.getElementById('content').style.opacity = 0;
                    
                    // Show the loading spinner
                    document.getElementById('loading').style.display = 'flex';
                    
                    // If this was triggered by the link, navigate after a short delay
                    if (event) {
                        setTimeout(function() {
                            window.location.href = '/default';
                        }, 100);
                    }
                    
                    return true;
                }
            </script>
        </body>
    </html>
    ''')

# Default data output
@app.route('/default')
def default():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    key = "cde91076b59b9a82d4244edda5f837d45a08"

    def run_clustering():
        logging.info("Running clustering...")
        try:
            PMIDs = open("./data/PMIDs_list.txt")
        except FileNotFoundError:
            logging.error("PMIDs_list.txt file not found.")
            raise

        id_datasets = {}

        for id in PMIDs:
            id_datasets[id[:-1]] = []
            url = f"{base_url}elink.fcgi?dbfrom=pubmed&db=gds&linkname=pubmed_gds&id={id}&api_key={key}&retmode=xml"
            try:
                ids = requests.get(url)
                ids.raise_for_status()
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching data for PMID {id}: {e}")
                continue

            root = ET.fromstring(ids.text)
            link_id_element = root.find(".//LinkSet/LinkSetDb/Link/Id")

            if link_id_element is not None:
                link_id = link_id_element.text
                id_datasets[id[:-1]].append(link_id)

        PMIDs.close()
        documents = {}
        for PMID, dataset in id_datasets.items():
            if not dataset:
                logging.warning(f"No dataset found for PMID {PMID}")
                continue
            url = f"{base_url}esummary.fcgi?db=gds&id={dataset[0]}&api_key={key}&retmode=xml&version=2.0"
            try:
                experiment = requests.get(url)
                experiment.raise_for_status()
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching data for dataset {dataset[0]}: {e}")
                continue
            root = ET.fromstring(experiment.text)
            title_root = root.find(".//DocumentSummarySet/DocumentSummary/title").text if root.find(".//DocumentSummarySet/DocumentSummary/title") is not None else ""
            summary_root = root.find(".//DocumentSummarySet/DocumentSummary/summary").text if root.find(".//DocumentSummarySet/DocumentSummary/summary") is not None else ""
            taxon_root = root.find(".//DocumentSummarySet/DocumentSummary/taxon").text if root.find(".//DocumentSummarySet/DocumentSummary/taxon") is not None else ""
            gdstype_root = root.find(".//DocumentSummarySet/DocumentSummary/gdstype").text if root.find(".//DocumentSummarySet/DocumentSummary/gdstype") is not None else ""
            document_info = summary_root + title_root + taxon_root + gdstype_root
            documents[PMID] = document_info

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(documents.values())
        distance_matrix = cosine_distances(vectors)

        dbscan = DBSCAN(eps=0.3, min_samples=2, metric='precomputed')
        cluster_labels = dbscan.fit_predict(distance_matrix)

        results = {}
        for i, label in enumerate(cluster_labels):
            results[i + 1] = {'PMID': list(id_datasets.keys())[i], 'Cluster': label}

        logging.info("Clustering completed.")
        return results, distance_matrix
    

    def visualize_clusters(results, distance_matrix):
        logging.info("Visualizing clusters...")
        G = nx.Graph()

        for node, info in results.items():
            G.add_node(info['PMID'], cluster=info['Cluster'])

        n = len(results)
        for i in range(n):
            for j in range(i + 1, n):
                if distance_matrix[i, j] < 0.5:
                    G.add_edge(list(results.values())[i]['PMID'], list(results.values())[j]['PMID'])

        pos = nx.spring_layout(G, seed=42)

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        node_text = []
        node_color = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f'PMID: {node}<br>Cluster: {G.nodes[node]["cluster"]}')
            node_color.append(G.nodes[node]["cluster"])

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=10,
                color=node_color,
                colorbar=dict(title='Cluster')
            ),
            text=node_text,
            hoverinfo='text'
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Cluster Visualization',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))

        logging.info("Visualization complete.")
        # return pyo.plot(fig, output_type='div')
        return fig

    results, distance_matrix = run_clustering()
    return visualize_clusters(results, distance_matrix).show()

#input data output
@app.route('/result', methods=['POST'])
def result():
    user_input = request.form.get('user_input', '')
    # return f"<h1>Вы ввели: {user_input}</h1>"

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    key = "cde91076b59b9a82d4244edda5f837d45a08"

    def run_clustering():
        logging.info("Running clustering...")
        PMIDs = [x.strip() for x in user_input.split(',') if x.strip()]

        id_datasets = {}

        for id in PMIDs:
            id_datasets[id[:-1]] = []
            url = f"{base_url}elink.fcgi?dbfrom=pubmed&db=gds&linkname=pubmed_gds&id={id}&api_key={key}&retmode=xml"
            try:
                ids = requests.get(url)
                ids.raise_for_status()
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching data for PMID {id}: {e}")
                continue

            root = ET.fromstring(ids.text)
            link_id_element = root.find(".//LinkSet/LinkSetDb/Link/Id")

            if link_id_element is not None:
                link_id = link_id_element.text
                id_datasets[id[:-1]].append(link_id)

        documents = {}
        for PMID, dataset in id_datasets.items():
            if not dataset:
                logging.warning(f"No dataset found for PMID {PMID}")
                continue
            url = f"{base_url}esummary.fcgi?db=gds&id={dataset[0]}&api_key={key}&retmode=xml&version=2.0"
            try:
                experiment = requests.get(url)
                experiment.raise_for_status()
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching data for dataset {dataset[0]}: {e}")
                continue
            root = ET.fromstring(experiment.text)
            title_root = root.find(".//DocumentSummarySet/DocumentSummary/title").text if root.find(".//DocumentSummarySet/DocumentSummary/title") is not None else ""
            summary_root = root.find(".//DocumentSummarySet/DocumentSummary/summary").text if root.find(".//DocumentSummarySet/DocumentSummary/summary") is not None else ""
            taxon_root = root.find(".//DocumentSummarySet/DocumentSummary/taxon").text if root.find(".//DocumentSummarySet/DocumentSummary/taxon") is not None else ""
            gdstype_root = root.find(".//DocumentSummarySet/DocumentSummary/gdstype").text if root.find(".//DocumentSummarySet/DocumentSummary/gdstype") is not None else ""
            document_info = summary_root + title_root + taxon_root + gdstype_root
            documents[PMID] = document_info

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(documents.values())
        distance_matrix = cosine_distances(vectors)

        dbscan = DBSCAN(eps=0.3, min_samples=2, metric='precomputed')
        cluster_labels = dbscan.fit_predict(distance_matrix)

        results = {}
        for i, label in enumerate(cluster_labels):
            results[i + 1] = {'PMID': list(id_datasets.keys())[i], 'Cluster': label}

        logging.info("Clustering completed.")
        return results, distance_matrix


    def visualize_clusters(results, distance_matrix):
        logging.info("Visualizing clusters...")
        G = nx.Graph()

        for node, info in results.items():
            G.add_node(info['PMID'], cluster=info['Cluster'])

        n = len(results)
        for i in range(n):
            for j in range(i + 1, n):
                if distance_matrix[i, j] < 0.5:
                    G.add_edge(list(results.values())[i]['PMID'], list(results.values())[j]['PMID'])

        pos = nx.spring_layout(G, seed=42)

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        node_text = []
        node_color = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f'PMID: {node}<br>Cluster: {G.nodes[node]["cluster"]}')
            node_color.append(G.nodes[node]["cluster"])

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=10,
                color=node_color,
                colorbar=dict(title='Cluster')
            ),
            text=node_text,
            hoverinfo='text'
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Cluster Visualization',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))

        logging.info("Visualization complete.")
        # return pyo.plot(fig, output_type='div')
        return fig

    results, distance_matrix = run_clustering()
    return visualize_clusters(results, distance_matrix).show()

if __name__ == '__main__':
    app.run(debug=True)