"""Train-only selection and bounded pair-wise logit corrections.

The original confidence/augmentation machinery is deliberately unchanged.
"""
from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def select_confusing_pairs(matrix, max_pairs=2, min_errors=2):
    n = len(matrix)
    candidates = []
    for a in range(n):
        for b in range(a+1, n):
            forward, backward = int(matrix[a][b]), int(matrix[b][a])
            if forward > 0 and backward > 0 and forward+backward >= min_errors:
                candidates.append((-(forward+backward), a, b))
    candidates.sort()
    return [(a,b) for _,a,b in candidates[:max_pairs]]


class PairDiscriminator(nn.Module):
    def __init__(self, input_dim, n_classes, hidden=64, max_pairs=2, strength=.5):
        super().__init__()
        if min(input_dim, hidden, max_pairs) < 1 or n_classes < 2:
            raise ValueError('positive dimensions and at least two classes required')
        if not math.isfinite(strength) or strength < 0:
            raise ValueError('pair strength must be finite and nonnegative')
        self.n_classes, self.max_pairs, self.strength = n_classes, max_pairs, strength
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
            for _ in range(max_pairs)])
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.register_buffer('pair_ids', torch.full((max_pairs,2), -1, dtype=torch.long))
        self.register_buffer('selection_done', torch.tensor(False))
        self.override_off = False
        self.last_scores = None

    def active_pairs(self):
        return [(int(a),int(b)) for a,b in self.pair_ids.detach().cpu().tolist() if a >= 0]

    def set_pairs(self, pairs):
        pairs = [(min(int(a),int(b)),max(int(a),int(b))) for a,b in pairs]
        if len(pairs) > self.max_pairs or len(set(pairs)) != len(pairs):
            raise ValueError('too many or duplicate pairs')
        if any(a < 0 or a == b or b >= self.n_classes for a,b in pairs):
            raise ValueError('pair labels must be distinct valid class IDs')
        self.pair_ids.fill_(-1)
        for i,(a,b) in enumerate(pairs):
            self.pair_ids[i] = torch.tensor([a,b],device=self.pair_ids.device)
        self.selection_done.fill_(True)

    def forward(self, features, base_logits, mask=None):
        shape = (*base_logits.shape[:-1], self.max_pairs)
        scores = base_logits.new_zeros(shape)
        gates, shifts = torch.zeros_like(scores), torch.zeros_like(scores)
        self.last_scores = None
        pairs = self.active_pairs()
        if not pairs or self.override_off:
            return base_logits, dict(base_logits=base_logits.detach(), gate=gates,
                                     shift=shifts, expert_score=scores)
        probability = base_logits.detach().softmax(-1)
        score_list, gate_list, shift_list = [], [], []
        correction = torch.zeros_like(base_logits)
        for index,(a,b) in enumerate(pairs):
            score = self.heads[index](features).squeeze(-1)
            mass = probability[...,a] + probability[...,b]
            # 0 when irrelevant/one-sided, <=1 when both labels are plausible.
            gate = (mass * 4 * probability[...,a] * probability[...,b]
                    / mass.square().clamp_min(1e-12)).detach()
            if mask is not None:
                gate = gate * mask.to(gate.dtype)
            shift = self.strength * torch.tanh(score) * gate
            correction[...,a] += shift
            correction[...,b] -= shift
            score_list.append(score); gate_list.append(gate); shift_list.append(shift)
        # Fixed-capacity caches keep CSV schemas stable across warm-up/selection.
        padding = self.max_pairs-len(pairs)
        self.last_scores = torch.stack(score_list,-1)
        scores = F.pad(self.last_scores, (0,padding))
        gates = F.pad(torch.stack(gate_list,-1), (0,padding))
        shifts = F.pad(torch.stack(shift_list,-1), (0,padding))
        return base_logits+correction, dict(base_logits=base_logits.detach(),
            gate=gates.detach(), shift=shifts.detach(), expert_score=scores.detach())

    def auxiliary_loss(self, labels, mask):
        if self.last_scores is None:
            return self.pair_ids.new_zeros((),dtype=torch.float32)
        labels = labels.reshape(mask.shape)
        losses = []
        for index,(a,b) in enumerate(self.active_pairs()):
            valid = mask.bool() & ((labels == a) | (labels == b))
            if valid.any():
                losses.append(F.binary_cross_entropy_with_logits(
                    self.last_scores[...,index][valid],(labels[valid] == a).float()))
        return torch.stack(losses).mean() if losses else self.last_scores.sum()*0


def add_pair_arguments(parser):
    parser.add_argument('--disable-pair', dest='use_pair',action='store_false',default=True)
    parser.add_argument('--pair-hidden',type=int,default=64)
    parser.add_argument('--pair-max-pairs',type=int,default=2)
    parser.add_argument('--pair-strength',type=float,default=.5)
    parser.add_argument('--pair-loss-weight',type=float,default=.05)
    parser.add_argument('--pair-warmup',type=int,default=5,
                        help='normal confidence epochs before clean TRAIN-only pair selection')
    parser.add_argument('--pair-min-errors',type=int,default=2)
    parser.add_argument('--pairs',nargs='+',default=None,metavar='A:B',
                        help='optional manual pair IDs; otherwise select from TRAIN predictions')


def validate_pair_arguments(parser,args):
    if min(args.pair_hidden,args.pair_max_pairs,args.pair_min_errors) < 1 or args.pair_warmup < 0:
        parser.error('positive pair dimensions/min-errors and nonnegative warmup required')
    if any(not math.isfinite(x) or x < 0 for x in (args.pair_strength,args.pair_loss_weight)):
        parser.error('pair strength/loss weight must be finite and nonnegative')
    if args.pairs:
        try:
            pairs = [tuple(map(int,x.split(':'))) for x in args.pairs]
            n = 6 if args.Dataset == 'IEMOCAP' else 7
            if any(len(p)!=2 or min(p)<0 or max(p)>=n or p[0]==p[1] for p in pairs):
                raise ValueError()
            normalized = [tuple(sorted(p)) for p in pairs]
            if len(normalized)>args.pair_max_pairs or len(set(normalized))!=len(normalized):
                raise ValueError()
        except (ValueError,TypeError):
            parser.error('--pairs expects distinct valid pairs such as 0:4 3:5 within capacity')


def validate_pair_checkpoint(checkpoint,args):
    saved = checkpoint.get('args',{})
    enabled = saved.get('use_pair',False)
    names = ['use_pair']
    if enabled and args.use_pair:
        names += ['pair_hidden','pair_max_pairs','pair_strength']
    mismatch = [name for name in names if saved.get(name,False if name=='use_pair' else None) != getattr(args,name)]
    if mismatch:
        raise ValueError('Pair checkpoint configuration mismatch: '+', '.join(mismatch)+
                         '. For an old confidence checkpoint use --disable-pair.')


@contextmanager
def preserved_evaluation(model):
    cpu = torch.get_rng_state()
    gpu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_rng = np.random.get_state()
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad(): yield
    finally:
        model.train(was_training)
        torch.set_rng_state(cpu)
        if gpu is not None: torch.cuda.set_rng_state_all(gpu)
        np.random.set_state(numpy_rng)


def select_pairs_from_training(model,loader,args,directory,epoch):
    head = model.pair_discriminator
    if head is None or bool(head.selection_done): return
    if args.pairs:
        head.set_pairs([tuple(map(int,p.split(':'))) for p in args.pairs])
        metadata = dict(source='manual_configuration',selection_epoch=epoch,
                        pairs=head.active_pairs(),note='User supplied; no evaluation labels queried.')
    else:
        # Rebuild an ordered loader using ONLY the original training indices.
        from torch.utils.data import DataLoader,Subset
        indices = getattr(loader.sampler,'indices',None)
        if indices is None:
            raise ValueError('TRAIN pair selection requires explicit training sampler indices')
        dataset = Subset(loader.dataset,sorted(map(int,indices)))
        ordered = DataLoader(dataset,batch_size=loader.batch_size,shuffle=False,num_workers=0,
                             collate_fn=loader.dataset.collate_fn)
        matrix = np.zeros((model.n_classes,model.n_classes),dtype=np.int64)
        device = next(model.parameters()).device
        with preserved_evaluation(model):
            for data in ordered:
                t,v,a,q,mask,label = [x.to(device) for x in data[:-1]]
                lengths = mask.sum(1).long().tolist()
                prediction = model(t,v,a,mask,q.permute(1,0,2),lengths)[4].argmax(-1)
                valid = mask.bool()
                np.add.at(matrix,(label[valid].cpu().numpy(),prediction[valid].cpu().numpy()),1)
        head.set_pairs(select_confusing_pairs(matrix.tolist(),args.pair_max_pairs,args.pair_min_errors))
        metadata = dict(source='clean_training_predictions',selection_epoch=epoch,
            training_utterances=int(matrix.sum()),training_dialogues=len(indices),
            rank='bidirectional error count; both directions must be nonzero',
            pairs=head.active_pairs(),matrix=matrix.tolist(),
            note='No test/validation predictions used. Selection frozen after this pass.')
    path = Path(directory)/'pair_selection.json'
    path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Pair selection ({}): {}'.format(metadata['source'],head.active_pairs() or 'none; no eligible pairs'))


def collect_pair_rows(model,mask,labels,dialogues):
    info = model.last_pair
    if info is None: return []
    base = info['base_logits'].log_softmax(-1).cpu()
    final = info['final_logits'].log_softmax(-1).cpu()
    gates,shifts = info['gate'].cpu(),info['shift'].cpu()
    pairs = model.pair_discriminator.active_pairs() if model.pair_discriminator is not None else []
    capacity = gates.size(-1)
    rows = []
    labels = labels.detach().cpu()
    for b,dialogue in enumerate(dialogues):
        for turn in mask[b].bool().nonzero(as_tuple=False).flatten().tolist():
            y = int(labels[b,turn])
            before,after = int(base[b,turn].argmax()),int(final[b,turn].argmax())
            row = dict(dialogue=str(dialogue),turn=turn,true_label=y,
                       base_prediction=before,prediction=after,
                       base_nll=-float(base[b,turn,y]),corrected_nll=-float(final[b,turn,y]),
                       corrected=int(before!=y and after==y),harmed=int(before==y and after!=y),
                       prediction_changed=int(before!=after))
            for i in range(capacity):
                a,c = pairs[i] if i<len(pairs) else (-1,-1)
                row.update({f'pair{i}_a':a,f'pair{i}_b':c,
                            f'pair{i}_gate':float(gates[b,turn,i]),
                            f'pair{i}_shift':float(shifts[b,turn,i])})
            rows.append(row)
    return rows


def pair_summary(rows):
    if not rows: return {}
    from sklearn.metrics import accuracy_score,f1_score
    y = [r['true_label'] for r in rows]
    base = [r['base_prediction'] for r in rows]
    final = [r['prediction'] for r in rows]
    return dict(utterances=len(rows),base_ACC=100*accuracy_score(y,base),
        corrected_ACC=100*accuracy_score(y,final),base_F1=100*f1_score(y,base,average='weighted'),
        corrected_F1=100*f1_score(y,final,average='weighted'),
        base_macro_F1=100*f1_score(y,base,average='macro'),
        corrected_macro_F1=100*f1_score(y,final,average='macro'),
        base_NLL=float(np.mean([r['base_nll'] for r in rows])),
        corrected_NLL=float(np.mean([r['corrected_nll'] for r in rows])),
        corrected=sum(r['corrected'] for r in rows),harmed=sum(r['harmed'] for r in rows),
        prediction_changed=sum(r['prediction_changed'] for r in rows))


def write_pair_best(directory,rows,model,epoch):
    from reliability import save_sample_scores
    if not rows: return
    directory = Path(directory)
    save_sample_scores(directory/'best_pair_predictions.csv',rows)
    summary = pair_summary(rows)
    pairs = model.pair_discriminator.active_pairs() if model.pair_discriminator else []
    summary.update(epoch=epoch,pairs=pairs,pair_confusions=summarize_pair_confusions(rows,pairs),
                   note='Same selected checkpoint, clean TEST input: before vs after logit correction; not a retrained ablation.')
    (directory/'best_pair_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')


def log_pair_epoch(directory,rows,epoch,model,verbose=False):
    from reliability import append_rows
    summary = pair_summary(rows)
    if not summary: return
    append_rows(Path(directory)/'pair_metrics.csv',[dict(epoch=epoch,**summary)])
    pairs = model.pair_discriminator.active_pairs() if model.pair_discriminator else []
    per_pair = summarize_pair_confusions(rows,pairs)
    if per_pair:
        append_rows(Path(directory)/'pair_confusion_metrics.csv',[dict(epoch=epoch,**r) for r in per_pair])
    if verbose:
        pairs = model.pair_discriminator.active_pairs() if model.pair_discriminator else []
        print('Pair TEST (current epoch {}): wF1 {:.2f} -> {:.2f} | corrected={} harmed={} | pairs={}'.format(
            epoch,summary['base_F1'],summary['corrected_F1'],summary['corrected'],summary['harmed'],pairs))


def summarize_pair_confusions(rows,pairs):
    result = []
    for a,b in pairs:
        count = lambda prediction,first,second: sum(r['true_label']==first and r[prediction]==second for r in rows)
        before_ab,before_ba = count('base_prediction',a,b),count('base_prediction',b,a)
        after_ab,after_ba = count('prediction',a,b),count('prediction',b,a)
        result.append(dict(label_a=a,label_b=b,
            support_a=sum(r['true_label']==a for r in rows),support_b=sum(r['true_label']==b for r in rows),
            base_a_to_b=before_ab,base_b_to_a=before_ba,base_pair_errors=before_ab+before_ba,
            corrected_a_to_b=after_ab,corrected_b_to_a=after_ba,corrected_pair_errors=after_ab+after_ba))
    return result
