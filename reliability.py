"""Utterance-level modality scores, paired feature degradation, and diagnostics.

These scores are learned proxies for feature reliability, not calibrated
probabilities of being correct. Modality order is always text/audio/visual.
"""
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MODALITIES = 'tav'


class UtteranceReliability(nn.Module):
    def __init__(self, input_dim, hidden=64, strength=0.5):
        super().__init__()
        if input_dim < 1 or hidden < 1 or not 0 <= strength <= 1:
            raise ValueError('positive dimensions and reliability strength in [0,1] required')
        self.strength = float(strength)
        self.heads = nn.ModuleDict({m: nn.Sequential(nn.Linear(input_dim, hidden),
            nn.GELU(), nn.Linear(hidden, 1)) for m in MODALITIES})
        for head in self.heads.values():
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def gains(self, scores):
        # Average gain remains 1. Equal scores reproduce the original fusion.
        # At strength=.5, gains lie in [.5,2], including numerical edge cases.
        safe = scores.clamp_min(1e-6)
        relative = 3 * safe / safe.sum(-1, keepdim=True)
        return 1 + self.strength * (relative - 1)

    def forward(self, features):
        logits = torch.cat([self.heads[m](features[m]) for m in MODALITIES], dim=-1)
        scores = logits.sigmoid()
        return logits, scores, self.gains(scores)

    def paired_loss(self, clean, degraded, mask, affected):
        clean_logits, clean_scores, _ = self(clean)
        bad_logits, bad_scores, _ = self(degraded)
        valid = mask.bool().unsqueeze(-1).expand_as(clean_logits)
        if not valid.any():
            return clean_logits.sum()*0, dict(bce=0., ranking=0.)
        affected = affected.bool() & valid
        clean_target = torch.full_like(clean_logits, .9)
        bad_target = torch.where(affected, torch.full_like(bad_logits, .1), clean_target)
        bce = .5 * (F.binary_cross_entropy_with_logits(clean_logits[valid], clean_target[valid]) +
                    F.binary_cross_entropy_with_logits(bad_logits[valid], bad_target[valid]))
        if affected.any():
            ranking = F.relu(.2 - clean_scores[affected] + bad_scores[affected]).mean()
        else:
            ranking = bad_logits.sum()*0
        loss = bce + .5 * ranking
        return loss, dict(bce=float(bce.detach()), ranking=float(ranking.detach()))


def training_feature_stds(dataset):
    """Streaming per-channel population std; only dataset.keys are inspected."""
    result = {}
    for m, attribute in [('t','videoText'),('a','videoAudio'),('v','videoVisual')]:
        total = squared = None
        count = 0
        for key in dataset.keys:
            values = np.asarray(getattr(dataset, attribute)[key], dtype=np.float64)
            if values.ndim != 2 or not np.isfinite(values).all():
                raise ValueError('training features must be finite utterance-by-channel arrays')
            if total is None:
                total = np.zeros(values.shape[1]); squared = np.zeros(values.shape[1])
            total += values.sum(0); squared += np.square(values).sum(0); count += len(values)
        if not count:
            raise ValueError('cannot estimate corruption scales from an empty training set')
        variance = np.maximum(squared/count - np.square(total/count), 0)
        result[m] = torch.from_numpy(np.maximum(np.sqrt(variance), 1e-6).astype(np.float32))
    return result


class FeatureCorruptor:
    def __init__(self, stds, device, seed=17, noise_min=.5, noise_max=1.5, zero_prob=.2):
        self.device = torch.device(device)
        self.stds = {m: stds[m].to(self.device) for m in MODALITIES}
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        if not 0 <= noise_min <= noise_max or not 0 <= zero_prob <= 1:
            raise ValueError('invalid feature corruption configuration')
        self.noise_min, self.noise_max, self.zero_prob = noise_min, noise_max, zero_prob

    def corrupt(self, features, mask, modalities='tav', strength=None, kind='mixed'):
        if not modalities or any(m not in MODALITIES for m in modalities):
            raise ValueError('select corruption modalities from t/a/v')
        if kind not in ('mixed','noise','zero'):
            raise ValueError('unknown feature corruption type')
        valid = mask.bool()
        choice = torch.randint(len(modalities), valid.shape, device=self.device, generator=self.generator)
        affected = torch.zeros(*valid.shape, 3, dtype=torch.bool, device=self.device)
        result = {}
        for index, m in enumerate(MODALITIES):
            source = features[m].transpose(0,1)
            selected = valid & (choice == modalities.index(m)) if m in modalities else torch.zeros_like(valid)
            if kind == 'zero':
                replacement = torch.zeros_like(source)
            else:
                noise = torch.randn(source.shape, device=self.device, dtype=source.dtype, generator=self.generator)
                if strength is None:
                    scale = self.noise_min + (self.noise_max-self.noise_min)*torch.rand(
                        *valid.shape, 1, device=self.device, generator=self.generator)
                else:
                    scale = float(strength)
                replacement = source + noise*self.stds[m]*scale
                if kind == 'mixed':
                    zero = torch.rand(valid.shape, device=self.device, generator=self.generator) < self.zero_prob
                    replacement = torch.where(zero.unsqueeze(-1), torch.zeros_like(replacement), replacement)
            # A zero vector zeroed again is unchanged: do not teach the head
            # contradictory labels for an identical clean/degraded pair.
            affected[:,:,index] = selected & (replacement != source).any(-1)
            result[m] = torch.where(selected.unsqueeze(-1), replacement, source).transpose(0,1)
        return result, affected

    def mix(self, clean, degraded, mask, probability):
        if not 0 <= probability <= 1:
            raise ValueError('corruption probability must be in [0,1]')
        selected = (torch.rand(mask.shape, device=self.device, generator=self.generator) < probability) & mask.bool()
        return {m: torch.where(selected.transpose(0,1).unsqueeze(-1), degraded[m], clean[m]) for m in MODALITIES}

    def state_dict(self):
        return {m: self.stds[m].detach().cpu() for m in MODALITIES}


def paired_reliability_loss(model, clean, degraded, mask, affected):
    if model.reliability is None:
        return next(model.parameters()).sum()*0, dict(bce=0., ranking=0.)
    projected = []
    # Paired supervision trains the small heads, not a projector that could
    # learn to manipulate the artificial clean/degraded distinction.
    with torch.no_grad():
        for features in (clean, degraded):
            projected.append({m: getattr(model, {'t':'textf_input','a':'acouf_input','v':'visuf_input'}[m])(
                features[m].permute(1,2,0)).transpose(1,2) for m in MODALITIES})
    return model.reliability.paired_loss(projected[0], projected[1], mask, affected)


def add_reliability_arguments(parser):
    parser.add_argument('--disable-reliability', dest='use_reliability', action='store_false', default=True)
    parser.add_argument('--reliability-hidden', type=int, default=64)
    parser.add_argument('--reliability-strength', type=float, default=.5)
    parser.add_argument('--reliability-loss-weight', type=float, default=.1)
    parser.add_argument('--corruption-prob', type=float, default=.25,
                        help='probability that a training utterance uses its degraded version')
    parser.add_argument('--corruption-modalities', choices=['t','a','v','ta','tv','av','tav'], default='tav')
    parser.add_argument('--noise-min', type=float, default=.5)
    parser.add_argument('--noise-max', type=float, default=1.5)
    parser.add_argument('--zero-prob', type=float, default=.2)
    parser.add_argument('--probe-every', type=int, default=10, help='0 disables periodic feature degradation probes')
    parser.add_argument('--data-dir', default=None)


def validate_reliability_arguments(parser, args):
    if args.reliability_hidden < 1 or not 0 <= args.reliability_strength <= 1:
        parser.error('positive reliability-hidden and strength in [0,1] required')
    if not np.isfinite(args.reliability_loss_weight) or args.reliability_loss_weight < 0:
        parser.error('reliability-loss-weight must be finite and nonnegative')
    if not 0 <= args.corruption_prob <= 1 or not 0 <= args.zero_prob <= 1:
        parser.error('corruption probabilities must be in [0,1]')
    if not np.isfinite(args.noise_max) or not 0 <= args.noise_min <= args.noise_max:
        parser.error('require finite 0 <= noise-min <= noise-max')
    if args.probe_every < 0:
        parser.error('probe-every must be >=0')


def validate_reliability_checkpoint(checkpoint, args):
    saved = checkpoint.get('args', {})
    enabled = saved.get('use_reliability', False)
    names = ['use_reliability'] + (['reliability_hidden','reliability_strength'] if enabled and args.use_reliability else [])
    mismatch = [name for name in names if saved.get(name, False) != getattr(args,name)]
    if mismatch:
        raise ValueError('Checkpoint reliability configuration mismatch: ' + ', '.join(mismatch))


def append_rows(path, rows):
    if not rows:
        return
    path = Path(path); exists = path.exists()
    with path.open('a', encoding='utf-8-sig' if not exists else 'utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists: writer.writeheader()
        writer.writerows(rows)


def save_sample_scores(path, rows):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def collect_score_rows(model, mask, labels, predicted, dialogue_ids):
    if model.last_reliability is None:
        return []
    scores, gains = [x.detach().cpu().numpy() for x in model.last_reliability]
    valid = mask.detach().cpu().numpy().astype(bool)
    labels = labels.detach().cpu().numpy()
    predicted = predicted.reshape(valid.shape)
    rows = []
    for b, turn in zip(*np.nonzero(valid)):
        row = dict(dialogue=str(dialogue_ids[b]),turn=int(turn),true_label=int(labels[b,turn]),
                   prediction=int(predicted[b,turn]))
        for index,m in enumerate(MODALITIES):
            row['score_'+m] = float(scores[b,turn,index]); row['gain_'+m] = float(gains[b,turn,index])
        rows.append(row)
    return rows


def run_reliability_probe(model, loader, stds, seed, epoch, directory, full=False):
    """Fixed feature-level interventions; not training or an automatic selector."""
    from sklearn.metrics import accuracy_score, f1_score
    device = next(model.parameters()).device
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type=='cuda' else None
    was_training = model.training
    old_override = model.reliability_override
    model.eval()
    conditions = [('clean',None,None,0.),('audio_noise','a','noise',1.),('visual_noise','v','noise',1.),
                  ('audio_zero','a','zero',0.),('visual_zero','v','zero',0.),('uniform_gates',None,None,0.)]
    if full:
        conditions += [('{}_noise_{}'.format(label,scale),m,'noise',scale)
                       for label,m in [('audio','a'),('visual','v')] for scale in (.5,2.)]
    rows = []
    try:
        for condition, modality, kind, noise_scale in conditions:
            model.reliability_override = 'uniform' if condition=='uniform_gates' else None
            corruptor = FeatureCorruptor(stds,device,seed=seed+701)
            labels, predictions, scores, gains = [],[],[],[]
            with torch.no_grad():
                for data in loader:
                    text,vision,audio,speaker,mask,target = [x.to(device) for x in data[:-1]]
                    features = dict(t=text,a=audio,v=vision)
                    if modality:
                        features,_ = corruptor.corrupt(features,mask,modalities=modality,kind=kind,strength=noise_scale)
                    outputs = model(features['t'],features['v'],features['a'],mask,speaker.permute(1,0,2),
                                    mask.sum(1).long().tolist())
                    valid = mask.bool()
                    labels.extend(target[valid].cpu().tolist())
                    predictions.extend(outputs[4].argmax(-1)[valid].cpu().tolist())
                    if model.last_reliability is not None:
                        q,g = model.last_reliability
                        scores.append(q[valid].cpu().numpy()); gains.append(g[valid].cpu().numpy())
            row = dict(epoch=epoch,condition=condition,noise_scale=noise_scale,
                       ACC=100*accuracy_score(labels,predictions),
                       weighted_F1=100*f1_score(labels,predictions,average='weighted',zero_division=0),
                       macro_F1=100*f1_score(labels,predictions,average='macro',zero_division=0))
            for index,m in enumerate(MODALITIES):
                row['score_'+m] = float(np.concatenate(scores)[:,index].mean()) if scores else float('nan')
                row['gain_'+m] = float(np.concatenate(gains)[:,index].mean()) if gains else 1.
            rows.append(row)
    finally:
        model.reliability_override = old_override
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None: torch.cuda.set_rng_state(cuda_rng,device)
    append_rows(Path(directory)/'robustness_probe.csv',rows)
    values = {row['condition']:row['weighted_F1'] for row in rows}
    print('Current-model probe wF1: clean={:.2f}% | A-noise={:.2f}% | V-noise={:.2f}%'.format(
        values['clean'],values['audio_noise'],values['visual_noise']))
    return rows
