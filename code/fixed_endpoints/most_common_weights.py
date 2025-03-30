import requests
from statistics import mode

SOURCES = [
    "/s/resource/wordnet", "/s/resource/dbpedia", "/s/resource/verbosity", 
    "/s/resource/wiktionary", "/s/resource/opencyc", "/s/contributor/omcs"
]

sources_most_common_weights = []

for source in SOURCES:
    offset = 0
    isEnd = False
    weights = []

    while not isEnd:
        obj = requests.get(f"http://api.conceptnet.io/{source}?offset={offset}&limit=1000").json()
        edges = obj["edges"]
        if len(edges) < 1000:
            isEnd = True
        for edge in edges:
            weights.append(edge["weight"])
        offset += 1000

    sources_most_common_weights.append(mode(weights))

print(f"Sources_most_common_weights = {sources_most_common_weights}")
