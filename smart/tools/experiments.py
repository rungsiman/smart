import argparse
import json


CONFIG_FILE = 'experiments.json'
OUTPUT_FILE = 'run'


def build(writer, config, choices):
    global build_datasets
    global do_not_clean
    global freezer_counter

    if len(config) > len(choices):
        for item in list(config.values())[len(choices)]:
            build(writer, config, choices + [item])

    else:
        params = ' '.join('--%s="%s"' % (list(config.keys())[i],
                                         str(choices[i]).replace(' ', '').replace("'", "\\\""))
                          for i in range(len(config)))

        for dataset in (['dbpedia', 'wikidata'] if build_datasets == 'all' else [build_datasets]):
            for task in ['literal', 'hybrid']:
                for stage in ['train', 'test']:
                    writer.write(f'bash {stage} {task} {dataset} {params}\n')

            writer.write('bash freeze --identifier="id-%04d-dependent" --exclude-models\n' % freezer_counter)

            for strategy in ['independent', 'top-down', 'bottom-up']:
                writer.write(f'bash test hybrid {dataset} {params} --all-hybrid-dataset-test_strategy={strategy}\n')
                writer.write(f'bash freeze --identifier="id-%04d-{strategy}" --exclude-models\n' % freezer_counter)

            if do_not_clean:
                writer.write('\n')
            else:
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', action='store', default='all')
    parser.add_argument('--do-not-clean', action='store_true')
    args = parser.parse_args()
    build_datasets = args.dataset
    do_not_clean = args.do_not_clean
    run()
