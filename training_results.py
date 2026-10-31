"""Human-readable, per-run training output. Does not change model selection."""
from datetime import datetime
from pathlib import Path
import re
import sys
import traceback

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


class _Tee:
    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile

    def write(self, text):
        self.logfile.write(text)
        self.logfile.flush()
        return self.terminal.write(text)

    def flush(self):
        self.logfile.flush()
        self.terminal.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


class TrainingRun:
    def __init__(self, script_path, args):
        self.args = args
        root = Path(args.result_dir) if args.result_dir else Path(script_path).resolve().parent / 'result'
        root.mkdir(parents=True, exist_ok=True)
        name = args.run_name or '{}_seed{}'.format(args.Dataset.lower(), args.seed)
        name = re.sub(r'[^A-Za-z0-9_.-]+', '_', name).strip('._') or 'run'
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.directory = (root / '{}_{}'.format(name, stamp)).resolve()
        base_directory = self.directory
        suffix = 1
        while True:
            try:
                self.directory.mkdir()
                break
            except FileExistsError:
                self.directory = base_directory.with_name(base_directory.name + '_{}'.format(suffix))
                suffix += 1

    def __enter__(self):
        self.logfile = (self.directory / 'training_log.txt').open('w', encoding='utf-8', buffering=1)
        self.stdout, self.stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(self.stdout, self.logfile)
        sys.stderr = _Tee(self.stderr, self.logfile)
        print('Result directory: {}'.format(self.directory))
        print('Started: {}'.format(datetime.now().isoformat(timespec='seconds')))
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is not None and exc_type is not SystemExit:
                traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
            print('Finished: {}'.format(datetime.now().isoformat(timespec='seconds')))
        finally:
            sys.stdout, sys.stderr = self.stdout, self.stderr
            self.logfile.close()
        return False

    def save_checkpoint(self, payload):
        import torch
        target = self.directory / 'best_model.pt'
        temporary = target.with_suffix('.tmp')
        # Python handles Unicode paths; older Windows PyTorch path writers do not.
        with temporary.open('wb') as file:
            torch.save(payload, file)
        temporary.replace(target)
        return target

    def log_details(self, text):
        """Keep full details in the live log without flooding the terminal."""
        self.logfile.write(str(text).rstrip('\n') + '\n')
        self.logfile.flush()

    def write_epoch(self, epoch, total_epochs, train_metrics, valid_metrics, test_metrics,
                    best_fscore, best_epoch, elapsed, is_best=False):
        train_loss, train_acc, train_fscore = train_metrics
        valid_loss, valid_acc, valid_fscore = valid_metrics
        test_loss, test_acc, test_fscore = test_metrics
        self.log_details(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, '
            'valid_loss: {}, valid_acc: {}, valid_fscore: {}, '
            'test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.format(
                epoch, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore,
                test_loss, test_acc, test_fscore, round(elapsed, 2)))
        print('Epoch {}/{} | Loss={:.4f} | ACC={:.2f}% | wF1={:.2f}% | Best={:.2f}%@{}{}'.format(
            epoch, total_epochs, train_loss, test_acc, test_fscore, best_fscore, best_epoch,
            ' *' if is_best else ''))

    def write_best(self, epoch, train_metrics, valid_metrics, test_metrics,
                   labels, predictions, mask, checkpoint_path, reliability_diagnostics=None,
                   train_loss_components=None):
        labels, predictions, mask = (np.asarray(x).reshape(-1) for x in (labels, predictions, mask))
        valid = mask > 0
        labels, predictions, weights = labels[valid], predictions[valid], mask[valid]
        classes = list(range(6 if self.args.Dataset == 'IEMOCAP' else 7))
        report = classification_report(labels, predictions, labels=classes, sample_weight=weights,
                                       output_dict=True, zero_division=0)
        matrix = confusion_matrix(labels, predictions, labels=classes, sample_weight=weights)
        lines = [
            'CSS Best Training Result',
            'Dataset: {}'.format(self.args.Dataset),
            'Selection rule: highest test weighted-F1; higher test ACC breaks ties.',
            'Best epoch: {}'.format(epoch),
            'Seed: {}'.format(self.args.seed),
            'Checkpoint: {}'.format(checkpoint_path),
            '', 'Metrics from this SAME selected epoch:',
        ]
        for split, metrics in [('train', train_metrics), ('valid', valid_metrics), ('test', test_metrics)]:
            if not np.isfinite(metrics[0]):
                lines.append('{} metrics: unavailable (empty validation set)'.format(split))
            else:
                lines.append('{}_loss: {:.4f}; {}_ACC (%): {:.2f}; {}_weighted_F1 (%): {:.2f}'.format(
                    split, metrics[0], split, metrics[1], split, metrics[2]))
        lines.extend([
            'Macro-F1 (%): {:.2f}'.format(report['macro avg']['f1-score'] * 100),
            'Valid test utterances: {}'.format(int(valid.sum())),
            '', 'Per-label TEST metrics (percent):',
            'Precision and recall retain their standard definitions; labels retain original numeric IDs.',
        ])
        for label in classes:
            item = report[str(label)]
            lines.append('Label {}: precision={:.2f}%, recall={:.2f}%, F1={:.2f}%, support={:g}'.format(
                label, item['precision'] * 100, item['recall'] * 100,
                item['f1-score'] * 100, item['support']))
        lines.extend(['', 'Classification report (fractions):',
                      classification_report(labels, predictions, labels=classes, sample_weight=weights,
                                            digits=4, zero_division=0),
                      'Confusion matrix (rows=true labels, columns=predicted labels):',
                      np.array2string(matrix, max_line_width=160), '',
                      'Train loss = original PGM loss + reliability_loss_weight * paired reliability loss + pair_loss_weight * pair BCE.',
                      'Test loss = original PGM loss only. Losses are not error percentages.',
                      'Train metrics were accumulated during parameter updates, not re-evaluated afterwards.',
                      'Configuration:', str(vars(self.args))])
        if train_loss_components is not None:
            lines.extend(['', 'Selected-epoch training loss components:', str(train_loss_components)])
        if reliability_diagnostics:
            lines.extend(['', 'Selected-epoch clean TEST reliability scores and gains:', str(reliability_diagnostics),
                          'Scores are feature-degradation proxies, not calibrated correctness probabilities.',
                          'See best_reliability.csv for per-utterance values.'])
        lines.extend(['', 'Pair discrimination:',
                      'See best_pair_summary.json and best_pair_predictions.csv for SAME-checkpoint correction effects.',
                      'Pairs selected once from clean TRAIN predictions after warm-up, or explicit manual configuration.',
                      'Saved pair corrections are an inference diagnostic, not a separately retrained ablation.'])
        target = self.directory / 'best_results.txt'
        temporary = target.with_suffix('.tmp')
        temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        temporary.replace(target)
