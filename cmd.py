import argparse
import json
from collections import defaultdict
from typing import List
import pandas as pd
import yaml
from path import Path
from paths import RUN, ROOT
from test import main as test_main, TestParams
from train import distance_f


def clean_empty_runs():
    """
    Delete experiments under "run/" which doesn't have a checkpoint
    """
    for run_folder in RUN.dirs():
        files_run: List[Path] = run_folder.files()
        if not any(x.basename().startswith('checkpoint') for x in files_run):
            print('to delete', repr(run_folder))
            run_folder.rmtree()


def run_evals(args):
    """
    Run evaluations on test set of all experiments using paper parameters.
    Evaluation parameters are stored in "eval_dataset_runs.yaml".
    Additional args are: --device and --batch-size
    """
    with open(ROOT / 'eval_dataset_runs.yaml') as f:
        runs_per_dataset = yaml.load(f)
    for run in RUN.dirs():
        if not any(f.basename().startswith('checkpoint') for f in run.files()):
            print('skip', run)
            continue
        with open(run / 'config.yaml') as f:
            train_config = yaml.safe_load(f)
        dataset: str = train_config['dataset']
        print('examining ', repr(run), '...', sep='')
        print('dataset', dataset)
        runs_params: List[dict] = runs_per_dataset[dataset]
        for run_params in runs_params:
            run_params = TestParams(**run_params,
                                    run_path=run,
                                    device=args.device,
                                    batch_size=args.batch_size,
                                    steps=600, dataset=dataset,
                                    distance=distance_f(train_config['distance']),
                                    train_config=train_config)
            print('run params:', run_params)
            test_main(run_params)


def make_table_eval():
    """
    Make table by gathering all experiment evaluation results
    """
    table = defaultdict(list)
    for run in RUN.dirs():
        print('load', str(run))
        eval_folder = run / 'evaluation'
        with open(run / 'config.yaml') as f:
            train_config = yaml.safe_load(f)
        train_dataset = train_config['dataset']
        train_query_samples = train_config['query_samples']
        train_support_samples = train_config['support_samples']
        train_num_classes = train_config['num_classes']
        if eval_folder.exists():
            for json_file in eval_folder.files('*.json'):
                with open(json_file) as f:
                    eval_results = json.load(f)
                eval_query_samples = eval_results['query_samples']
                eval_support_samples = eval_results['support_samples']
                eval_num_classes = eval_results['num_classes']
                eval_accuracy = eval_results['avg_accuracy']
                eval_loss = eval_results['avg_loss']
                table['Dataset'].append(train_dataset)
                table['Train num classes'].append(train_num_classes)
                table['Train support samples'].append(train_support_samples)
                table['Train query samples'].append(train_query_samples)
                table['Test num classes'].append(eval_num_classes)
                table['Test support samples'].append(eval_support_samples)
                table['Test query samples'].append(eval_query_samples)
                table['Test accuracy'].append(eval_accuracy)
                table['Test loss'].append(eval_loss)
    table = pd.DataFrame(table)
    aggr_table = table.pivot_table(columns=table.columns.drop(['Test accuracy', 'Test loss']).tolist(),
                                   aggfunc={'Test accuracy': 'max', 'Test loss': 'min'})
    print(aggr_table.T)
    aggr_table.T.to_csv('eval_results.csv')


cmds = {
    'clean_empty_runs': [clean_empty_runs],
    'make_table_eval': [make_table_eval],
    'run_evals': [run_evals,
                  [['--device'], dict(type=str, required=True, dest='device')],
                  [['--batch-size'], dict(type=int, required=True, dest='batch_size')]
    ]
}

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    subparsers = argument_parser.add_subparsers(title='cmd', dest='cmd')
    for cmd in cmds:
        values = cmds[cmd]
        func = values[0]
        subparser = subparsers.add_parser(cmd, help=func.__doc__)
        for i in range(1, len(values)):
            names = values[i][0]
            arg_kwargs = values[i][1]
            subparser.add_argument(*names, **arg_kwargs)

    args = argument_parser.parse_args()
    cmd_args = cmds[args.cmd]
    funct = cmd_args[0]
    if len(cmd_args) > 1:
        funct(args)
    else:
        funct()
