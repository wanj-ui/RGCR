import os

from torch.optim.lr_scheduler import ReduceLROnPlateau


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np, argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset, resolve_data_path
from model import MaskedNLLLoss, MaskedKLDivLoss, Transformer_Based_Model, PGM
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import pickle as pk
import datetime
from training_results import TrainingRun
from reliability import (add_reliability_arguments, validate_reliability_arguments,
    validate_reliability_checkpoint, training_feature_stds, FeatureCorruptor,
    paired_reliability_loss, collect_score_rows, save_sample_scores,
    append_rows, run_reliability_probe)
from pair_discrimination import (add_pair_arguments, validate_pair_arguments, validate_pair_checkpoint,
    select_pairs_from_training, collect_pair_rows, write_pair_best, log_pair_epoch)
torch.backends.cudnn.enabled = False
#
def set_seed(seed=61078):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

def get_train_valid_sampler(trainset, valid=0.1, dataset='MELD'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False, data_dir=None):
    trainset = MELDDataset(resolve_data_path('meld_multimodal_features.pkl',data_dir))
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset(resolve_data_path('meld_multimodal_features.pkl',data_dir), train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False, data_dir=None):
    trainset = IEMOCAPDataset(data_dir=data_dir)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False,data_dir=data_dir)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, kl_loss, dataloader, epoch, optimizer=None, train=False):
    losses, preds, labels, masks = [], [], [], []
    pgm_losses, quality_losses, sample_scores = [], [], []
    pair_losses, pair_samples = [], []
    PGM_loss = PGM(loss_function, kl_loss)

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        quality_loss = None
        if train:
            clean = dict(t=textf, a=acouf, v=visuf)
            degraded, affected = corruptor.corrupt(clean, umask, modalities=args.corruption_modalities)
            training_features = corruptor.mix(clean, degraded, umask, args.corruption_prob)
            if args.use_reliability:
                quality_loss, _ = paired_reliability_loss(model, clean, degraded, umask, affected)
            textf, acouf, visuf = [training_features[m] for m in 'tav']
        outputs = model(textf, visuf, acouf, umask, qmask, lengths)
        labels_ = label.view(-1)


        shared_params = [

            *model.shared_transformer.parameters(),

            *model.textf_input.parameters(),
            *model.acouf_input.parameters(),
            *model.visuf_input.parameters(),


            *model.t_t.parameters(),
            *model.a_a.parameters(),
            *model.v_v.parameters(),


            *model.t_t_gate.parameters(),
            *model.t_gate.parameters(),
            *model.a_a_gate.parameters(),
            *model.a_gate.parameters(),
            *model.v_v_gate.parameters(),
            *model.v_gate.parameters(),


            *model.features_reduce_t.parameters(),
            *model.features_reduce_a.parameters(),
            *model.features_reduce_v.parameters(),

        ]

        loss, weights, individual_losses = PGM_loss(
            outputs,
            labels_,
            umask,
            shared_params
        )

        pgm_loss_value = float(loss.detach())
        pair_loss = (model.pair_discriminator.auxiliary_loss(label,umask)
                     if train and model.pair_discriminator is not None else None)
        pair_losses.append((float(pair_loss.detach()) if pair_loss is not None else 0.0)*float(umask.sum()))
        if pair_loss is not None:
            loss = loss + args.pair_loss_weight * pair_loss
        quality_loss_value = float(quality_loss.detach()) if quality_loss is not None else 0.0
        if quality_loss is not None:
            loss = loss + args.reliability_loss_weight * quality_loss
        valid_count = float(umask.sum())
        pgm_losses.append(pgm_loss_value * valid_count)
        quality_losses.append(quality_loss_value * valid_count)
        all_prob = outputs[4]
        pred_ = torch.argmax(all_prob.view(-1, all_prob.size()[2]), 1)

        preds.append(pred_.data.cpu().numpy())
        if not train:
            sample_scores.extend(collect_score_rows(model,umask,label,preds[-1],data[-1]))
            pair_samples.extend(collect_pair_rows(model,umask,label,data[-1]))
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item() * masks[-1].sum())

        if train:
            loss.backward()
            if args.tensorboard:

                for param in model.named_parameters():
                    if param[1].grad is not None:
                        writer.add_histogram(param[0], param[1].grad, epoch)


                writer.add_scalar('weights/task1_weight', weights[0], epoch)
                writer.add_scalar('weights/task2_weight', weights[1], epoch)
                writer.add_scalar('weights/task3_weight', weights[2], epoch)


                for i, task_loss in enumerate(individual_losses):
                    writer.add_scalar(f'losses/task{i + 1}_loss', task_loss.item(), epoch)
                writer.add_scalar('losses/total_loss', loss.item(), epoch)

            optimizer.step()


    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        model.last_epoch_details = dict(pgm_loss=float('nan'),reliability_loss=float('nan'),pair_loss=float('nan'),samples=[],pair_samples=[])
        return float('nan'), float('nan'), [], [], [], float('nan')

    model.last_epoch_details = dict(pgm_loss=float(np.sum(pgm_losses)/np.sum(masks)),
                                    reliability_loss=float(np.sum(quality_losses)/np.sum(masks)),
                                    pair_loss=float(np.sum(pair_losses)/np.sum(masks)),
                                    samples=sample_scores,pair_samples=pair_samples)
    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)

    # plot_tsne(outputs[9], labels_)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore


def evaluate_loaded_model(model, dataloader):
    preds, labels, masks, pair_samples = [], [], [], []
    model.eval()

    with torch.no_grad():
        for data in dataloader:
            textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
            qmask = qmask.permute(1, 0, 2)
            lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

            outputs = model(textf, visuf, acouf, umask, qmask, lengths)
            all_prob = outputs[4]
            pred_ = torch.argmax(all_prob.view(-1, all_prob.size()[2]), 1)

            preds.append(pred_.data.cpu().numpy())
            labels.append(label.view(-1).data.cpu().numpy())
            pair_samples.extend(collect_pair_rows(model,umask,label,data[-1]))
            masks.append(umask.view(-1).cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    masks = np.concatenate(masks)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)

    model.last_eval_pair_rows = pair_samples
    return avg_accuracy, labels, preds, masks, avg_fscore


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=1024, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')
    parser.add_argument('--epochs', type=int, default=200, metavar='E', help='number of epochs')
    parser.add_argument('--temp', type=int, default=1, metavar='temp', help='temp')
    parser.add_argument('--rank', type=int, default=16, metavar='rank', help='projection rank')
    parser.add_argument('--order', type=int, default=3, metavar='order', help='fusion order')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')
    parser.add_argument('--seed', type=int, default=11, help='random seed')
    parser.add_argument('--result-dir', '--save-best-dir', dest='result_dir', default=None, help='result root (default: result/ beside this script); each run gets its own subdirectory')
    parser.add_argument('--run-name', default=None, help='checkpoint name prefix')
    parser.add_argument('--checkpoint', default=None, help='checkpoint path to load')
    parser.add_argument('--eval-only', action='store_true', default=False, help='only evaluate a loaded checkpoint')

    add_reliability_arguments(parser)
    add_pair_arguments(parser)
    args = parser.parse_args()
    validate_reliability_arguments(parser,args)
    validate_pair_arguments(parser,args)
    if args.run_name is None:
        variant = 'confidence' if args.use_reliability else 'augmentation_control'
        if not args.use_reliability and args.corruption_prob == 0: variant = 'baseline_control'
        variant += '_pair' if args.use_pair else '_pair_off'
        args.run_name = '{}_{}_seed{}'.format(args.Dataset.lower(),variant,args.seed)
    with TrainingRun(__file__, args) as results:
        set_seed(args.seed)
        today = datetime.datetime.now()
        if args.run_name is None:
            args.run_name = '{}_seed{}'.format(args.Dataset.lower(), args.seed)
        filename = 'iemocap_multimodal_features.pkl' if args.Dataset == 'IEMOCAP' else 'meld_multimodal_features.pkl'
        args.data_dir = str(resolve_data_path(filename,args.data_dir).parent)
        results.log_details(args)
        print('Variant: UTTERANCE_RELIABILITY | enabled={} | head={} | train corruption={:.0f}%'.format(
            args.use_reliability,args.reliability_hidden,100*args.corruption_prob))
        print('Pair discrimination: enabled={} | max_pairs={} | warmup={} | selection=clean TRAIN only'.format(
            args.use_pair,args.pair_max_pairs,args.pair_warmup))

        args.cuda = torch.cuda.is_available() and not args.no_cuda
        if args.cuda:
            print('Running on GPU')
        else:
            print('Running on CPU')

        if args.tensorboard:
            from tensorboardX import SummaryWriter

            writer = SummaryWriter()

        cuda = args.cuda
        n_epochs = args.epochs
        batch_size = args.batch_size
        feat2dim = {'IS10': 1582, 'denseface': 342, 'MELD_audio': 300}
        D_audio = feat2dim['IS10'] if args.Dataset == 'IEMOCAP' else feat2dim['MELD_audio']
        D_visual = feat2dim['denseface']
        D_text = 1024

        D_m = D_audio + D_visual + D_text

        n_speakers = 9 if args.Dataset == 'MELD' else 2
        n_classes = 7 if args.Dataset == 'MELD' else 6 if args.Dataset == 'IEMOCAP' else 1

        print('temp {}'.format(args.temp))

        model = Transformer_Based_Model(args.Dataset, args.temp, D_text, D_visual, D_audio, args.n_head,args.rank,args.order,
                                        n_classes=n_classes,
                                        hidden_dim=args.hidden_dim,
                                        n_speakers=n_speakers,
                                        dropout=args.dropout,
                                        use_reliability=args.use_reliability,
                                        reliability_hidden=args.reliability_hidden,
                                        reliability_strength=args.reliability_strength,
                                        use_pair=args.use_pair,pair_hidden=args.pair_hidden,
                                        pair_max_pairs=args.pair_max_pairs,pair_strength=args.pair_strength)

        total_params = sum(p.numel() for p in model.parameters())
        print('total parameters: {}'.format(total_params))
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('training parameters: {}'.format(total_trainable_params))

        if cuda:
            model.cuda()

        kl_loss = MaskedKLDivLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)



        if args.Dataset == 'MELD':

            loss_function = MaskedNLLLoss()

            train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                       batch_size=batch_size,
                                                                       num_workers=0, data_dir=args.data_dir)
        elif args.Dataset == 'IEMOCAP':
            loss_weights = torch.FloatTensor([1 / 0.086747,
                                              1 / 0.144406,
                                              1 / 0.227883,
                                              1 / 0.160585,
                                              1 / 0.127711,
                                              1 / 0.252668])

            loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)

            train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                          batch_size=batch_size,
                                                                          num_workers=0, data_dir=args.data_dir)
        else:
            print("There is no such dataset")

        feature_stds = training_feature_stds(train_loader.dataset)
        corruptor = FeatureCorruptor(feature_stds,'cuda' if cuda else 'cpu',seed=args.seed+10001,
                                     noise_min=args.noise_min,noise_max=args.noise_max,zero_prob=args.zero_prob)

        if args.checkpoint is not None:
            map_location = 'cuda' if cuda else 'cpu'
            checkpoint = torch.load(args.checkpoint, map_location=map_location)
            validate_reliability_checkpoint(checkpoint,args)
            validate_pair_checkpoint(checkpoint,args)
            model_state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(model_state_dict)
            print('Loaded checkpoint: {}'.format(args.checkpoint))

        if args.eval_only:
            if args.checkpoint is None:
                raise ValueError('--eval-only requires --checkpoint')

            test_acc, test_label, test_pred, test_mask, test_fscore = evaluate_loaded_model(model, test_loader)
            write_pair_best(results.directory,model.last_eval_pair_rows,model,checkpoint.get('epoch',0))
            log_pair_epoch(results.directory,model.last_eval_pair_rows,checkpoint.get('epoch',0),model,verbose=True)
            print('Checkpoint test performance..')
            print('Acc: {}'.format(test_acc))
            print('F-Score: {}'.format(test_fscore))
            print(classification_report(test_label, test_pred, sample_weight=test_mask, digits=4, zero_division=0))
            print(confusion_matrix(test_label, test_pred, sample_weight=test_mask))

            if args.probe_every:
                run_reliability_probe(model,test_loader,feature_stds,args.seed,checkpoint.get('epoch',0),results.directory,full=True)
            if args.tensorboard:
                writer.close()
            exit()

        best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
        best_acc = None
        all_fscore, all_acc, all_loss = [], [], []

        for e in range(n_epochs):
            start_time = time.time()
            if args.use_pair and (args.pairs or e >= args.pair_warmup):
                select_pairs_from_training(model,train_loader,args,results.directory,e+1)

            train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(model, loss_function, kl_loss, train_loader,
                                                                               e, optimizer, True)
            train_details = model.last_epoch_details.copy()
            valid_loss, valid_acc, _, _, _, valid_fscore = train_or_eval_model(model, loss_function, kl_loss, valid_loader,
                                                                               e)
            test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(model, loss_function,
                                                                                                     kl_loss, test_loader,
                                                                                                         e)
            test_details = model.last_epoch_details.copy()
            all_fscore.append(test_fscore)

            is_best = False
            if best_fscore == None or best_fscore <= test_fscore:
                if best_fscore == None:
                    best_fscore = test_fscore
                    best_acc = test_acc
                    best_label, best_pred, best_mask = test_label, test_pred, test_mask
                    is_best = True

                elif best_acc < test_acc and best_fscore == test_fscore:
                    best_acc = test_acc
                    best_fscore = test_fscore
                    best_label, best_pred, best_mask = test_label, test_pred, test_mask
                    is_best = True

                elif best_fscore < test_fscore:
                    best_acc = test_acc
                    best_fscore = test_fscore
                    best_label, best_pred, best_mask = test_label, test_pred, test_mask
                    is_best = True

            if is_best:
                best_epoch = e + 1
                best_metrics = {
                    'train': (train_loss, train_acc, train_fscore),
                    'valid': (valid_loss, valid_acc, valid_fscore),
                    'test': (test_loss, test_acc, test_fscore),
                }
                checkpoint_path = str(results.directory / 'best_model.pt')
                results.save_checkpoint({
                    'epoch': e + 1,
                    'best_acc': best_acc,
                    'best_fscore': best_fscore,
                    'metrics': best_metrics,
                    'feature_stds': {m:v.cpu() for m,v in feature_stds.items()},
                    'train_loss_components': {k:v for k,v in train_details.items() if k not in ('samples','pair_samples')},
                    'best_loss': test_loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': vars(args).copy(),
                    'dataset': args.Dataset,
                    'seed': args.seed,
                    'selected_pairs': model.pair_discriminator.active_pairs() if model.pair_discriminator else [],
                    'classification_report': classification_report(
                        best_label,
                        best_pred,
                        sample_weight=best_mask,
                        digits=4, zero_division=0
                    ),
                    'confusion_matrix': confusion_matrix(
                        best_label,
                        best_pred,
                        sample_weight=best_mask
                    ).tolist(),
                })
                results.log_details('Saved best checkpoint: {}'.format(checkpoint_path))
                if test_details['samples']:
                    save_sample_scores(results.directory/'best_reliability.csv',test_details['samples'])
                write_pair_best(results.directory,test_details['pair_samples'],model,best_epoch)
                results.write_best(
                    epoch=best_epoch,
                    train_metrics=best_metrics['train'],
                    valid_metrics=best_metrics['valid'],
                    test_metrics=best_metrics['test'],
                    labels=best_label, predictions=best_pred, mask=best_mask,
                    checkpoint_path=checkpoint_path,
                    train_loss_components={k:v for k,v in train_details.items() if k not in ('samples','pair_samples')},
                    reliability_diagnostics=({key:float(np.mean([row[key] for row in test_details['samples']]))
                        for key in ('score_t','score_a','score_v','gain_t','gain_a','gain_v')}
                        if test_details['samples'] else None),
                )

            if args.tensorboard:
                writer.add_scalar('test: accuracy', test_acc, e)
                writer.add_scalar('test: fscore', test_fscore, e)
                writer.add_scalar('train: accuracy', train_acc, e)
                writer.add_scalar('train: fscore', train_fscore, e)


            results.write_epoch(
                epoch=e+1,total_epochs=n_epochs,
                train_metrics=(train_loss,train_acc,train_fscore),
                valid_metrics=(valid_loss,valid_acc,valid_fscore),
                test_metrics=(test_loss,test_acc,test_fscore),
                best_fscore=best_fscore,best_epoch=best_epoch,
                elapsed=time.time()-start_time,is_best=is_best)
            row = dict(epoch=e+1,train_loss=train_loss,train_pgm_loss=train_details['pgm_loss'],
                       train_reliability_loss=train_details['reliability_loss'],
                       train_pair_loss=train_details['pair_loss'],train_ACC=train_acc,
                       train_F1=train_fscore,test_loss=test_loss,test_ACC=test_acc,test_F1=test_fscore,
                       best_F1=best_fscore,best_epoch=best_epoch)
            for m in 'tav':
                row['test_score_'+m] = (float(np.mean([s['score_'+m] for s in test_details['samples']]))
                                       if test_details['samples'] else float('nan'))
            append_rows(results.directory/'training_metrics.csv',[row])
            log_pair_epoch(results.directory,test_details['pair_samples'],e+1,model,verbose=(e+1)%10==0)
            if (e+1)%10 == 0:
                print('Best TEST report so far (selected epoch {}):'.format(best_epoch))
                print(classification_report(best_label,best_pred,sample_weight=best_mask,digits=4,zero_division=0))
                print('Confusion matrix (rows=true, columns=predicted):')
                print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))
            if args.probe_every and (e+1)%args.probe_every == 0:
                run_reliability_probe(model,test_loader,feature_stds,args.seed,e+1,results.directory)

        if args.tensorboard:
            writer.close()

        print('Best result: epoch={} | ACC={:.2f}% | wF1={:.2f}%'.format(best_epoch,best_acc,best_fscore))
        print('Best result summary: {}'.format(results.directory / 'best_results.txt'))

        print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4, zero_division=0))
        print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))
