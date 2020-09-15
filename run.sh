#!/bin/sh

echo "Experiment on DBpedia"
./train literal dbpeidia
./train hybrid dbpeidia
./test literal dbpeidia
./test hybrid dbpeidia

echo "Experiment on Wikidata"
./train literal wikidata
./train hybrid wikidata
./test literal wikidata
./test hybrid wikidata

bash freeze --exclude-models
