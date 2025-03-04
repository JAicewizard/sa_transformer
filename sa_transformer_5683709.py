'''
    The original Transformer code is from: http://nlp.seas.harvard.edu/2018/04/03/attention.html
'''

import os,sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
from lifelines import *
import pandas as pd
import  argparse
from operator import itemgetter
from concordance import concordance_index

print("FDSF", torch.version.cuda)
sys.stdout.flush()

parser = argparse.ArgumentParser(description='Survival analysis')
parser.add_argument('--max_time', type=int, default=500, help='Max number of months')
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--N', type=int, default=6, help='Number of modules')
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--d_ff', type=int, default=256)
parser.add_argument('--train_batch_size', type=int, default=16)
parser.add_argument('--drop_prob', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--coeff', type=float, default=1.0)
parser.add_argument('--coeff2', type=float, default=1.0)
parser.add_argument('--data_dir', type=str, default='data/split1')
parser.add_argument('--save_ckpt_dir', type=str, default='checkpoints')
parser.add_argument('--report_interval', type=int, default=5)
parser.add_argument('--data_parallel', action='store_true', help='use data parallel?')
parser.add_argument('--pred_method', type=str, choices=['mean', 'median'], default='mean')
parser.add_argument('--cont', type=bool)
opt = parser.parse_args()
print(opt)

# Create the checkpoint directory
if not os.path.exists(opt.save_ckpt_dir):
    os.makedirs(opt.save_ckpt_dir)

class TranDataset(Dataset):
    def __init__(self, features, labels, is_train=True):
        self.is_train = is_train
        self.data = []
        temp = []
        for feature, label in zip(features, labels):
            feature = torch.from_numpy(np.array(feature)).float()
            duration, is_observed = label[0], label[1]
            temp.append([duration, is_observed, feature])
        sorted_temp = sorted(temp, key=itemgetter(0))

        if self.is_train:
            new_temp = sorted_temp
        else:
            new_temp = temp

        for duration, is_observed, feature in new_temp:
            if int(duration) > opt.max_time:
                raise Exception("duration is ", int(duration))
            if is_observed:
                mask = opt.max_time * [1.]
                label = int(duration) * [1.] + (opt.max_time - int(duration)) * [0.]
                feature = torch.stack(opt.max_time * [feature])
                #self.data.append([feature, torch.tensor(duration).float(), torch.tensor(mask).float(), torch.tensor(label), torch.tensor(is_observed).byte()])
                self.data.append([feature.cuda(), torch.tensor(duration).float().cuda(), torch.tensor(mask).float().cuda(), torch.tensor(label).cuda(), torch.tensor(is_observed).byte().cuda()])
            else:
                # NOTE plus 1 to include day 0
                mask = int(duration + 1) * [1.] + (opt.max_time - int(duration + 1)) * [0.]
                label = opt.max_time * [1.]
                feature = torch.stack(opt.max_time * [feature])
                #self.data.append([feature, torch.tensor(duration).float(), torch.tensor(mask).float(), torch.tensor(label), torch.tensor(is_observed).byte()])
                self.data.append([feature.cuda(), torch.tensor(duration).float().cuda(), torch.tensor(mask).float().cuda(), torch.tensor(label).cuda(), torch.tensor(is_observed).byte().cuda()])

    def __getitem__(self, index_a):
        if self.is_train:
            if index_a == len(self.data) - 1:
                index_b = np.random.randint(len(self.data))
            else:
                # NOTE self.data is sorted
                index_b = np.random.randint(index_a+1, len(self.data))
            #print("AA", len(self.data[index_a]))
            #print("AA", self.data[index_a][0][0])
            return [ [self.data[index_a][i], self.data[index_b][i]] for i in range(len(self.data[index_a])) ]
        else:
            return self.data[index_a]

    def __len__(self):
        return len(self.data)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
        nbatches = query.size(0)
        
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -torch.tensor(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# Initial embedding for raw input
class SrcEmbed(nn.Module):
    def __init__(self, input_dim, d_model):
        super(SrcEmbed, self).__init__()
        self.w = nn.Linear(input_dim, d_model)
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.w(x))


# Final layer for the Transformer
class TranFinalLayer(nn.Module):
    def __init__(self, d_model):
        super(TranFinalLayer, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model // 2)
        self.norm = LayerNorm(d_model // 2)
        self.w_2 = nn.Linear(d_model // 2, 1)

    def forward(self, x):
        x = F.relu(self.w_1(x))
        x = self.norm(x)
        x = self.w_2(x)
        return torch.sigmoid(x.squeeze(-1))


class Encoder(nn.Module):
    def __init__(self, layer, N, d_model, dropout, num_features):
        super(Encoder, self).__init__()
        self.src_embed = SrcEmbed(num_features, d_model)
        self.position_encode = PositionalEncoding(d_model, dropout)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.final_layer = TranFinalLayer(d_model)
        
    def forward(self, x, mask=None):
        #print(x.shape)
        x = self.position_encode(self.src_embed(x))
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_layer(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def evaluate(encoder, test_loader):
    encoder.eval()
    with torch.no_grad():
        pred_durations, true_durations, is_observed = [], [], []
        pred_obs_durations, true_obs_durations = [], []

        total_surv_probs = []

        # NOTE batch size is 1
        for features, durations, mask, label, is_observed_single in test_loader:
            is_observed.append(is_observed_single[0].cpu())
            sigmoid_preds = encoder.forward(features)
            surv_probs = torch.cumprod(sigmoid_preds, dim=1).squeeze()
            total_surv_probs.append(surv_probs)

            if opt.pred_method == 'mean':
                pred_duration = torch.sum(surv_probs).item()
            elif opt.pred_method == 'median':
                pred_duration = 0
                while True:
                    if surv_probs[pred_duration] < 0.5:
                        break
                    else:
                        pred_duration += 1
                        if pred_duration == len(surv_probs):
                            break

            true_duration = durations.squeeze().item()
            pred_durations.append(pred_duration)
            true_durations.append(true_duration)

            if is_observed_single:
                pred_obs_durations.append(pred_duration)
                true_obs_durations.append(true_duration)

        total_surv_probs = torch.stack(total_surv_probs)

        pred_obs_durations = np.asarray(pred_obs_durations)
        true_obs_durations = np.asarray(true_obs_durations)
        mae_obs = np.mean(np.abs(pred_obs_durations - true_obs_durations))

        pred_durations = np.asarray(pred_durations)
        true_durations = np.asarray(true_durations)
        
        is_observed = np.asarray(is_observed, dtype=bool)

        print('pred durations OBS', pred_durations[is_observed].round())
        print('true durations OBS', true_durations[is_observed].round())

        print('pred durations CRS', pred_durations[~is_observed].round())
        print('true durations CRS', true_durations[~is_observed].round())

        test_cindex = concordance_index(true_durations, pred_durations, is_observed)
        print('c index', test_cindex, 'mean abs error (OBS)', mae_obs)
        sys.stdout.flush()
    return test_cindex, mae_obs, total_surv_probs


def checkpoint(model, total_surv_probs):
    datadir = opt.data_dir.replace('/', '.')
    model_out_path = "{}/best_model_trainset_{}.pth".format(opt.save_ckpt_dir, datadir)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

    # Save npy array
    surv_probs_out_path = "{}/best_surv_probs_test_{}.pth".format(opt.save_ckpt_dir, datadir)
    np.save(surv_probs_out_path, total_surv_probs.cpu().numpy())


def train(features, labels, encoder):
    train_loader = DataLoader(TranDataset(features[0], labels[0], is_train=True), batch_size=opt.train_batch_size, shuffle=True)
    # NOTE VAL batch size is 1
    val_loader = DataLoader(TranDataset(features[1], labels[1], is_train=False), batch_size=1, shuffle=False)
    # NOTE TEST batch size is 1
    test_loader = DataLoader(TranDataset(features[2], labels[2], is_train=False), batch_size=1, shuffle=False)
    # NOTE using C index as early stopping criterion
    best_val_cindex, best_test_cindex, best_test_mae, best_epoch = -1, -1, 9999999, 0

    optimizer = torch.optim.Adam(encoder.parameters(), lr=opt.lr)

    for t in range(opt.num_epochs):
        print('epoch', t)
        encoder.train()

        tot_loss = 0.
        n = 0
        for features, true_durations, mask, label, is_observed in train_loader:
            if n%10 == 0:
                print(n)
            n = n+1
            optimizer.zero_grad()

            is_observed_a = is_observed[0]
            mask_a = mask[0]
            mask_b = mask[1]
            label_a = label[0]
            label_b = label[1]
            true_durations_a = true_durations[0]
            true_durations_b = true_durations[1]

            #print(features[0])
            sigmoid_a = encoder.forward(features[0])
            surv_probs_a = torch.cumprod(sigmoid_a, dim=1)
            #print(sigmoid_a)
            #print(surv_probs_a)
            #print(mask_a)
            #print(label_a) 
            loss = nn.BCELoss()(surv_probs_a * mask_a, label_a * mask_a)

            sigmoid_b = encoder.forward(features[1])
            surv_probs_b = torch.cumprod(sigmoid_b, dim=1)

            cond_a = is_observed_a & (true_durations_a < true_durations_b)
            if torch.sum(cond_a) > 0:
                mean_lifetimes_a = torch.sum(surv_probs_a, dim=1)
                mean_lifetimes_b = torch.sum(surv_probs_b, dim=1)
                diff = mean_lifetimes_b[cond_a] - mean_lifetimes_a[cond_a]
                true_diff = true_durations_b[cond_a] - true_durations_a[cond_a]
                loss += opt.coeff * torch.mean(nn.ReLU()(true_diff - diff))

            cond_a2 = is_observed_a
            if torch.sum(cond_a2) > 0:
                mean_lifetimes_a = torch.sum(surv_probs_a, dim=1)
                loss += opt.coeff2 * F.l1_loss(mean_lifetimes_a[cond_a2], true_durations_a[cond_a2])

            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            sys.stdout.flush()
        print('train total loss', tot_loss)

        # Evaluate
        if t > 0 and t % opt.report_interval == 0:
            print('VAL')
            val_cindex, val_mae, val_total_surv_probs = evaluate(encoder, val_loader)
            print('TEST')
            test_cindex, test_mae, test_total_surv_probs = evaluate(encoder, test_loader)

            if val_cindex > best_val_cindex:
                best_val_cindex = val_cindex
                best_val_mae = val_mae
                best_test_cindex = test_cindex
                best_test_mae = test_mae
                best_epoch = t
                checkpoint(encoder, test_total_surv_probs)

            print('current val cindex', val_cindex, 'val mae', val_mae)
            print('BEST val cindex', best_val_cindex, 'mae', best_val_mae, 'at epoch', best_epoch)
            print('its test cindex', best_test_cindex, 'mae', best_test_mae)
            sys.stdout.flush()


def main(features, labels, num_features):
    print(opt.d_model)
    c = copy.deepcopy
    attn = MultiHeadedAttention(opt.num_heads, opt.d_model, opt.drop_prob)
    ff = PositionwiseFeedForward(opt.d_model, opt.d_ff, opt.drop_prob)
    encoder_layer = EncoderLayer(opt.d_model, c(attn), c(ff), opt.drop_prob)
    encoder = Encoder(encoder_layer, opt.N, opt.d_model, opt.drop_prob, num_features).cuda()
    if opt.data_parallel:
        encoder = torch.nn.DataParallel(encoder).cuda()
    if opt.cont:
        datadir = opt.data_dir.replace('/', '.')
        model_out_path = "{}/best_model_trainset_{}.pth".format(opt.save_ckpt_dir, datadir)
        encoder = torch.load(model_out_path, weights_only=False)
    train(features, labels, encoder)


def cat2num(df):
    df['conditioning_intensity'] = df['conditioning_intensity'].map({
    'NMA': 1, 
    'RIC': 2,
    'MAC': 3,
    'TBD': None,
    'No drugs reported': None,
    'N/A, F(pre-TED) not submitted': None})
    
    df['tbi_status'] = df['tbi_status'].map({
    'No TBI': 0, 
    'TBI +- Other, <=cGy': 1,
    'TBI +- Other, -cGy, fractionated': 2,
    'TBI + Cy +- Other': 3,
    'TBI +- Other, -cGy, single': 4,
    'TBI +- Other, >cGy': 5,
    'TBI +- Other, unknown dose': None})
    
    df['dri_score'] = df['dri_score'].map({
    'Low': 1, 
    'Intermediate': 2,
    'Intermediate - TED AML case <missing cytogenetics': 3,
    'High': 4,
    'High - TED AML case <missing cytogenetics': 5,
    'Very High': 6,
    'N/A - pediatric': -3,
    'N/A - non-malignant indication': -1,
    'TBD cytogenetics': -2,
    'N/A - disease not classifiable': -4,
    'Missing disease status': 0})
    
    df['cyto_score'] = df['cyto_score'].map({
    'Poor': 4,
    'Normal': 3,
    'Intermediate': 2,
    'Favorable': 1,
    'TBD': -1,
    'Other': -2,
    'Not tested': None})
    
    df['cyto_score_detail'] = df['cyto_score_detail'].map({
    'Poor': 3, 
    'Intermediate': 2,
    'Favorable': 1,
    'TBD': -1,
    'Not tested': None})
   
    return df

def fill_hla_combined_low(row):
    if np.isnan(row['hla_combined_low']): 
        components = [
            row['hla_match_drb1_low'], row['hla_match_dqb1_low'], 
            row['hla_match_a_low'], row['hla_match_b_low'], row['hla_match_c_low']
        ]
        if all([not np.isnan(x) for x in components]):
            return sum(components)
        else:
            if not np.isnan(row['hla_low_res_8']) and not np.isnan(row['hla_match_dqb1_low']):
                return row['hla_low_res_8'] + row['hla_match_dqb1_low']
            elif not np.isnan(row['hla_low_res_6']): 
                components_6 = [
                    row['hla_match_dqb1_low'], row['hla_match_c_low']
                ]
                if all([not np.isnan(x) for x in components_6]):
                    return row['hla_low_res_6'] + sum(components_6)
                else: 
                    return sum([x for x in components if not np.isnan(x)])
    return row['hla_combined_low']

hct_ci_mapping = {
    "arrhythmia": {"No": 0, "Not done": 0, "Yes": 1},  
    "cardiac": {"No": 0, "Not done": 0, "Yes": 1}, 
    "diabetes": {"No": 0, "Not done": 0, "Yes": 1},  
    "hepatic_mild": {"No": 0, "Not done": 0, "Yes": 1},
    "hepatic_severe": {"No": 0, "Not done": 0, "Yes": 3},
    "psych_disturb": {"No": 0, "Not done": 0, "Yes": 1}, 
    "obesity": {"No": 0, "Not done": 0, "Yes": 1}, 
    "rheum_issue": {"No": 0, "Not done": 0, "Yes": 2},
    "peptic_ulcer": {"No": 0, "Not done": 0, "Yes": 2},  
    "renal_issue": {"No": 0, "Not done": 0, "Yes": 2}, 
    "prior_tumor": {"No": 0, "Not done": 0, "Yes": 3}, 
    "pulm_moderate": {"No": 0, "Not done": 0, "Yes": 2}, 
    "pulm_severe": {"No": 0, "Not done": 0, "Yes": 3},  
}
def calculate_hct_ci_score(row, mapping):
        """
        This function calculates the hct_ci score
    
        Args:
            row (pd.Series): Patient Clinical Data
            mapping (dict): HCT-CI score mapping
    
        Returns:
            int: HCT-CI score
        """
    
        score = 0
    
        if "hepatic_severe" in row and row["hepatic_severe"] == "Yes":
            score += mapping["hepatic_severe"]["Yes"]
        elif "hepatic_mild" in row and row["hepatic_mild"] == "Yes":
            score += mapping["hepatic_mild"]["Yes"]
        if "pulm_moderate" in row and row["pulm_moderate"] == "Yes":
            score += mapping["pulm_moderate"]["Yes"]
        elif "pulm_severe" in row and row["pulm_severe"] == "Yes":
            score += mapping["pulm_severe"]["Yes"]
    
        # Other Conditions
        for condition, mapping_values in mapping.items():
            if condition not in ["hepatic_mild", "hepatic_severe","pulm_moderate", "pulm_severe"] and condition in row:
                score += mapping_values.get(row[condition], 0)
    
        return score

def add_features(df):
    #df["hct_ci_score"] = df.apply(lambda row: calculate_hct_ci_score(row, hct_ci_mapping), axis=1)
    #df['donor_recipient_age_diff'] = abs(df['donor_age'] - df['age_at_hct'])
    df = cat2num(df)
    df['hla_combined_low'] = df['hla_low_res_10']
    df['hla_combined_low'] = df.apply(fill_hla_combined_low, axis=1)
    #df['hla_match_ratio'] = (df['hla_high_res_8'] + df['hla_low_res_8']) / 16
    #df['years_since_2000'] = df['year_hct'] - 2000
    #df['null_count'] = df.isnull().sum(axis=1)
    #df['ci_score_danger'] = df['hct_ci_score'].apply(lambda x: 2 if x >= 3 else 1 if x >= 1 else 0)
    df['year_hct'] -= 2000
    return df

if __name__ == '__main__':
    train_features = add_features(pd.read_csv(opt.data_dir + '/train_features.csv', header=0, index_col=False))
    val_features = add_features(pd.read_csv(opt.data_dir + '/val_features.csv', header=0, index_col=False))
    test_features = add_features(pd.read_csv(opt.data_dir + '/test_features.csv', header=0, index_col=False))

    combined = pd.concat([train_features,val_features, test_features],axis=0,ignore_index=True)
    combined = combined.drop(columns=["ID", "vent_hist"])
    
    RMV = ["ID","efs","efs_time","y"]
    FEATURES = [c for c in combined.columns if not c in RMV]
    CATS = []
    for c in FEATURES:
        if train_features[c].dtype=="object":
            CATS.append(c)
            combined[c] = combined[c].fillna("NAN")

    for c in FEATURES:
        # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
        if c in CATS:
            print(f"{c}, ", end="")
            combined[c],_ = combined[c].factorize()
            combined[c] -= combined[c].min()
            combined[c] = combined[c].astype("int32")
            combined[c] = combined[c].astype("category")
        
        # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
        else:
            if combined[c].dtype=="float64":
                combined[c] = combined[c].astype("float32")
            if combined[c].dtype=="int64":
                combined[c] = combined[c].astype("int32")

    for i in combined.columns[combined.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
        combined[i] = combined[i].fillna(combined[i].mean())

    train_features = combined.iloc[:len(train_features)]
    val_features = combined.iloc[len(train_features):len(train_features)+len(val_features)]
    test_features = combined.iloc[len(train_features)+len(val_features):]
    features = [train_features.to_numpy(), val_features.to_numpy(), test_features.to_numpy()]

    train_labels = pd.read_csv(opt.data_dir + '/train_labels.csv', header=0, index_col=False).to_numpy()
    val_labels = pd.read_csv(opt.data_dir + '/val_labels.csv', header=0, index_col=False).to_numpy()
    test_labels = pd.read_csv(opt.data_dir + '/test_labels.csv', header=0, index_col=False).to_numpy()
    labels = [train_labels, val_labels, test_labels]

    num_features = train_features.shape[1]
    print('train features shape', train_features.shape)
    print('train labels shape', train_labels.shape)
    print('val features shape', val_features.shape)
    print('val labels shape', val_labels.shape)
    print('test features shape', test_features.shape)
    print('test labels shape', test_labels.shape)
    print()

    print('num features', num_features)
    total_data = train_features.shape[0] + val_features.shape[0] + test_features.shape[0]
    print('total data', total_data)
    print()

    print('train max label', train_labels[:,0].max())
    print('train min label', train_labels[:,0].min())
    print('val max label', val_labels[:,0].max())
    print('val min label', val_labels[:,0].min())
    print('test max label', test_labels[:,0].max())
    print('test min label', test_labels[:,0].min())

    main(features, labels, num_features)
