import argparse
import json


CONFIG_FILE = 'experiments.json'
OUTPUT_FILE = 'run'


def build(writer, config, choices):
    global build_dataset
    global build_pipeline
    global build_stage
    global do_not_clean
    global freezer_counter

    if len(config) > len(choices):
        for item in list(config.values())[len(choices)]:
            build(writer, config, choices + [item])

    else:
        params = ' '.join('--%s="%s"' % (list(config.keys())[i],
                                         str(choices[i]).replace(' ', '').replace("'", "\\\""))
                          for i in range(len(config)))

        for dataset in (['dbpedia', 'wikidata'] if build_dataset == 'all' else [build_dataset]):
            if build_pipeline == 'literal':
                if build_stage in ['train', 'all']:
                    writer.write(f'bash train literal {dataset} {params}\n')

                if build_stage in ['test', 'all']:
                    writer.write(f'bash test literal {dataset} {params}\n')

                writer.write(f'bash freeze --identifier="id-%04d-literal" --exclude-models\n' % freezer_counter)

            else:
                for pipeline in (['literal', 'hybrid'] if build_pipeline == 'all' else ['hybrid']):
                    for stage in (['train', 'test'] if build_stage == 'all' else [build_stage]):
                        writer.write(f'bash {stage} {pipeline} {dataset} {params}\n')

                if build_stage == 'train':
                    writer.write('bash freeze --identifier="id-%04d-hybrid" --exclude-models\n' % freezer_counter)

                else:
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
    configs = json.load(open(CONFIG_FILE))

    with open(OUTPUT_FILE, 'w') as writer:
        writer.write('#!/bin/bash\n\n')

        for config in configs:
            choices = []
            build(writer, config, choices)


if __name__ == '__main__':
    freezer_counter = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', action='store', default='all')
    parser.add_argument('--pipeline', dest='pipeline', action='store', default='all')
    parser.add_argument('--stage', dest='stage', action='store', default='all')
    parser.add_argument('--do-not-clean', action='store_true')
    args = parser.parse_args()
    build_dataset = args.dataset
    build_pipeline = args.pipeline
    build_stage = args.stage
    do_not_clean = args.do_not_clean
    run()
