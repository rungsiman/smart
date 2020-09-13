import json
import math
import pymongo
import requests
import sys
import time
from tqdm import tqdm

from smart.data.base import Ontology

URL = 'http://rungsiman-pro.local:8890/sparql'
PREFIX = """
         PREFIX dbo: <http://dbpedia.org/ontology/>
         PREFIX dbp: <http://dbpedia.org/property/>
         """
LIMIT = 10000000
SKIP_UNTIL = None


def retrieve_dbpedia(*, collection, cls, target, predicate, filters=''):
    def shorten_uri(uri):
        uri = uri.replace('http://dbpedia.org/resource/', 'dbr:')
        uri = uri.replace('http://dbpedia.org/property/', 'dbp:')
        uri = uri.replace('http://dbpedia.org/ontology/', 'dbo:')
        return uri

    def build_param(query):
        return {'default-graph-uri': 'http://dbpedia.org',
                'query': query,
                'format': 'application/sparql-results+json',
                'CXML_redir_for_subjs': 'CXML_redir_for_hrefs',
                'timeout': 30000,
                'debug': 'on',
                'run': '+Run+Query+'}

    print(f'Querying {target} of {cls}...')
    sys.stdout.flush()

    query_template = PREFIX + """
                         SELECT {task} WHERE {
                             SELECT ?subject ?%s FROM <http://dbpedia.org> WHERE {
                                 ?subject a %s .
                                 ?subject %s ?%s .
                                 %s
                             }
                             ORDER BY ?subject
                         }
                         """ % (target, cls, predicate, target, filters)

    query_count = query_template.replace('{task}', 'COUNT(*)').replace('ORDER BY ?subject', '')
    response = requests.get(URL, params=build_param(query_count))
    count = json.loads(response.text)
    num_expect = int(count['results']['bindings'][0]['callret-0']['value'])
    num_steps = int(math.ceil(num_expect / LIMIT))
    num_rows = 0

    print(f'Retrieving {num_expect} entries...')
    sys.stdout.flush()
    time.sleep(0.1)

    for i in tqdm(range(num_steps), f'{cls}=>{target}'):
        query = query_template.replace('{task}', f'?subject ?{target}') + """
                    LIMIT %d
                    OFFSET %d
                    """ % (LIMIT, i * LIMIT)

        response = requests.get(URL, params=build_param(query))

        data = json.loads(response.text)
        num_rows += len(data['results']['bindings'])

        for item in data['results']['bindings']:
            uri = shorten_uri(item['subject']['value'])
            value = shorten_uri(item[target]['value'])

            document = collection.find_one({'uri': uri})

            if document is None:
                document = {'uri': uri, 'label': [], 'type': [], 'same_as': []}
                document[target].append(value)
                collection.insert_one(document)

            else:
                if value not in document[target]:
                    if target == 'label':
                        document[target].append(value)
                    elif target == 'type' and value.startswith('dbo:'):
                        document[target].append(value)
                    elif target == 'same_as' and any(value.startswith(key) for key in ('dbo:', 'dbp:', 'dbr:', 'http://dbpedia.org/')):
                        document[target].append(value)

                collection.update_one({'uri': uri}, {
                    '$set': {
                        'label': document['label'],
                        'type': document['type'],
                        'same_as': document['same_as']
                    }
                })

    print(f'Registered {num_rows} instances of class {cls}')
    sys.stdout.flush()


def main(dataset):
    ontology = Ontology(f'../../data/input/{dataset}/{dataset}_types.tsv')
    client = pymongo.MongoClient()
    db = client['smart']
    collection = db['entities']
    collection.create_index([('uri', pymongo.ASCENDING)], unique=True)
    skipping = SKIP_UNTIL is not None

    for i, label in enumerate(ontology.labels.keys()):
        if skipping and label != SKIP_UNTIL:
            continue

        skipping = False

        if dataset == 'dbpedia':
            print(f'Querying dbpedia class {i + 1} of {len(ontology.labels)}')
            retrieve_dbpedia(collection=collection, cls=label, target='label',
                             predicate='rdfs:label|foaf:name|dbo:title|dbp:label',
                             filters='FILTER (lang(?label) = "en")')

            retrieve_dbpedia(collection=collection, cls=label, target='type', predicate='a')
            retrieve_dbpedia(collection=collection, cls=label, target='same_as', predicate='owl:sameAs')


if __name__ == '__main__':
    main('dbpedia')
