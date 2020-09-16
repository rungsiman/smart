import json


CONFIG_FILE = 'experiments.json'
OUTPUT_FILE = 'run'


def build(writer, config, choices):
    global freezer_counter

    if len(config) > len(choices):
        for item in list(config.values())[len(choices)]:
            build(writer, config, choices + [item])

    else:
        for dataset in ['dbpedia', 'wikidata']:
            for task in ['literal', 'hybrid']:
                for stage in ['train', 'test']:
                    writer.write(f'bash {stage} {task} {dataset} %s\n' %
                                 ' '.join('--%s="%s"' % (list(config.keys())[i], str(choices[i]).replace(' ', '')) for i in range(len(config))))

            writer.write('bash freeze --identifier="id-%03d" --exclude-models\n' % freezer_counter)
            writer.write('bash clean\n\n')
            freezer_counter += 1


def run():
    config = json.load(open(CONFIG_FILE))
    choices = []

    with open(OUTPUT_FILE, 'w') as writer:
        writer.write('#!/bin/bash\n\n')
        build(writer, config, choices)


if __name__ == '__main__':
    freezer_counter = 1
    run()
