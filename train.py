import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_LAUNCH_BLOCKING"] = "2"  # 让报错的位置更加准确
import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedNLLLoss, LSTMModel, GRUModel, Model, MaskedMSELoss, FocalLoss
from model_hyper import My_anchor_Loss, SupConLoss
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
import pandas as pd
import pickle as pk
import datetime
import ipdb
import torch.nn.functional as F


seed = 1475  # We use seed = 1475 on IEMOCAP and seed = 67137 on MELD

class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"⚠️ EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = model.state_dict()
            self.counter = 0

    # 在 EarlyStopping 的 save_checkpoint 方法中添加
    def save_checkpoint(self, model, epoch, score):
        """保存模型checkpoint"""
        # 创建目录（如果不存在）
        os.makedirs("./saved_models", exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_score': self.best_score,
            'args': vars(args)  # 保存训练参数
        }
        torch.save(checkpoint, f'./saved_models/checkpoint_epoch{epoch}_fscore{score:.4f}.pth.tar')
        print(f"💾 Model saved at epoch {epoch} with score {score:.4f}")

def label_smoothing_loss(pred, target, smoothing=0.1):
    confidence = 1.0 - smoothing
    logprobs = F.log_softmax(pred, dim=-1)
    nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = confidence * nll_loss + smoothing * smooth_loss
    return loss.mean()


def get_fine_supcontrastive_loss(cuda, aware_features, batch_label, diag_len, speker_seq):
    # aware_features = data['aware_tensor']
    # batch_label = data['label_tensor']
    # diag_len = data['text_len_tensor']
    # speker_seq = data['speaker_tensor']

    diag_index = []
    spk_index = []
    for ndiag in range(len(diag_len)):
        num_diag = diag_len[ndiag]
        diag_index.append(ndiag * torch.ones(num_diag))
        spk_index.append(speker_seq[ndiag][:num_diag])
    diag_index = torch.cat(diag_index).long()
    spk_index = torch.cat(spk_index)

    diag_sample = [str(i) + str(j) + str(k) for i, j, k in
                   zip(diag_index.tolist(), spk_index.tolist(), batch_label.tolist())]

    sample_label = []
    label_dict = {}

    dlabel = 0
    for item in diag_sample:
        if item not in label_dict:
            label_dict[item] = dlabel
            dlabel += 1
        sample_label.append(label_dict[item])

    # sample_label = torch.tensor(sample_label).cuda()
    # # print("Features shape:", aware_features.unsqueeze(1).shape)
    # # print("Labels shape:", sample_label.shape)
    # # 计算每个原始样本对应的视图数
    # n_views = aware_features.shape[0] // len(sample_label)  # 534/178=3
    #
    # # 复制标签以匹配特征数量
    # expanded_labels = sample_label.repeat(n_views)
    '''源码'''
    # loss = SupConLoss(features=torch.cat(aware_features, dim=0).unsqueeze(1),
    #                   labels=sample_label.repeat(len(aware_features)))
    # print("aware_features.shape:", aware_features.shape)
    # print("sample_label.shape:", sample_label.shape)
    sample_label = torch.tensor(sample_label, dtype=torch.long).to(aware_features.device)

    # 自动计算视图数，防止写死
    num_views = aware_features.shape[0] // sample_label.shape[0]

    # 正确扩展标签
    sample_label = sample_label.repeat(num_views)

    # 确保维度一致
    assert sample_label.shape[0] == aware_features.shape[0], \
        f"Mismatch: labels {sample_label.shape[0]} vs features {aware_features.shape[0]}"

    loss = SupConLoss(features=aware_features.unsqueeze(1),# [534, 1, 512]
                      labels=sample_label) # [534]
    return loss


def extract_model_features(model, dataloader, cuda, modals, model_path, dataset_name, model_type="proposed"):
    """
    专门用于提取模型特征的函数
    """
    """检查模型文件是否存在并打印信息"""
    print(f"🔍 Looking for model: {model_path}")
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")

        # 检查 saved_models 目录中的文件
        model_dir = './saved_models/'
        if os.path.exists(model_dir):
            print("📂 Available model files:")
            for file in os.listdir(model_dir):
                if file.endswith('.pth.tar'):
                    file_path = os.path.join(model_dir, file)
                    file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
                    print(f"   - {file} ({file_size:.2f} MB)")

        return None, None, None
        print(f"✅ Model file found: {model_path}")
        # 尝试在 saved_models 目录下查找
        possible_paths = [
            model_path,
            f'./saved_models/{model_path}',
            f'./{model_path}',
            f'../{model_path}'
        ]

        found = False
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                found = True
                break

        if not found:
            # 列出可用的模型文件
            print("❌ Model checkpoint not found. Available models:")
            if os.path.exists('./saved_models'):
                for file in os.listdir('./saved_models'):
                    if file.endswith('.pth.tar'):
                        print(f"   - ./saved_models/{file}")
            return None, None, None

    # 加载模型
    try:
        checkpoint = torch.load(model_path, map_location='cuda' if cuda else 'cpu')

        # 检查数据集是否匹配
        if 'dataset' in checkpoint and checkpoint['dataset'] != dataset_name:
            print(
                f"❌ Dataset mismatch! Checkpoint trained on {checkpoint['dataset']}, but current dataset is {dataset_name}")
            print("💡 Please train a new model or use the correct dataset")
            return None, None, None

        # 检查args中的数据集信息
        if 'args' in checkpoint and hasattr(checkpoint['args'], 'Dataset'):
            if checkpoint['args']['Dataset'] != dataset_name:
                print(
                    f"❌ Dataset mismatch in args! Checkpoint trained on {checkpoint['args']['Dataset']}, but current dataset is {dataset_name}")
                return None, None, None

        # 加载模型
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model state dict")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded checkpoint directly")

        print(f"✅ Model successfully loaded from {model_path}")
        if 'epoch' in checkpoint:
            print(f"   Epoch: {checkpoint['epoch']}")
        if 'best_score' in checkpoint:
            print(f"   Best score: {checkpoint['best_score']:.4f}")
        if 'fscore' in checkpoint:
            print(f"   F-score: {checkpoint['fscore']:.4f}")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

    model.eval()

    print(f"🔍 Extracting features for {model_type} model on {dataset_name}...")

    # 创建一个简单的损失函数占位符
    class DummyLoss:
        def __call__(self, *args, **kwargs):
            return torch.tensor(0.0).to(next(model.parameters()).device)

    class DummyCenterLoss:
        def __call__(self, *args, **kwargs):
            return torch.tensor(0.0).to(next(model.parameters()).device), None

    # 使用修改后的函数提取特征
    try:
        results = train_or_eval_graph_model(
            model, DummyLoss(), DummyCenterLoss(), dataloader, 0, cuda, modals,
            optimizer=None, train=False, dataset=dataset_name, extract_features=True
        )

        # 解包结果
        if len(results) == 13:  # 确保有13个返回值
            (avg_loss, avg_accuracy, labels, preds, avg_fscore, avg_micro_fscore,
             avg_macro_fscore, ei, et, en, el, final_features, aware_features) = results
        else:
            print(f"❌ Unexpected number of return values: {len(results)}")
            return None, None, None

    except Exception as e:
        print(f"❌ Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

    # 保存特征
    os.makedirs("./saved_features", exist_ok=True)

    if final_features is not None and len(final_features) > 0:
        np.save(f'./saved_features/{model_type}_{dataset_name}_final_features.npy', final_features)
        print(f"💾 Final features saved: {final_features.shape}")
    else:
        print("⚠️ No final features to save")
        final_features = None

    if aware_features is not None and len(aware_features) > 0:
        np.save(f'./saved_features/{model_type}_{dataset_name}_aware_features.npy', aware_features)
        print(f"💾 Aware features saved: {aware_features.shape}")
    else:
        print("⚠️ No aware features to save")
        aware_features = None

    if labels is not None and len(labels) > 0:
        np.save(f'./saved_features/{model_type}_{dataset_name}_labels.npy', labels)
        print(f"💾 Labels saved: {labels.shape}")
    else:
        print("⚠️ No labels to save")
        labels = None

    return final_features, aware_features, labels

# seed =  100 # 69.99
def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _init_fn(worker_id):
    np.random.seed(int(seed) + worker_id)


def get_train_valid_sampler(trainset, valid=0.1, dataset='IEMOCAP'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset('MELD_features/MELD_features_raw1.pkl')
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

    testset = MELDDataset('MELD_features/MELD_features_raw1.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    # trainset = IEMOCAPDataset()
    trainset = IEMOCAPDataset('IEMOCAP_features/IEMOCAP_features.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory, worker_init_fn=_init_fn)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    # testset = IEMOCAPDataset(train=False)
    testset = IEMOCAPDataset('IEMOCAP_features/IEMOCAP_features.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory, worker_init_fn=_init_fn)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    """

    """
    losses, preds, labels, masks = [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        max_sequence_len.append(textf.size(0))

        log_prob, alpha, alpha_f, alpha_b, _ = model(textf, qmask, umask)
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])
        labels_ = label.view(-1)
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_, 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids]


def train_or_eval_graph_model(model, loss_function, Multi_Center_Loss, dataloader, epoch, cuda, modals, optimizer=None, train=False,
                              dataset='IEMOCAP', extract_features=False):
    """
    新增参数 extract_features: 是否提取特征
    """
    losses, preds, labels = [], [], []
    scores, vids = [], []
    all_final_features = []  # 新增：存储最终特征
    all_aware_features = []  # 新增：存储共识感知特征

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    # 初始化特征变量
    final_features = None
    aware_features = None

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    # 设定梯度累积步数
    # accumulationsteps = 2
    # accumulation_co_unter = 0
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf1, textf2, textf3, textf4, visuf, acouf, qmask, umask, label = [d.cuda() for d in
                                                                             data[:-1]] if cuda else data[:-1]

        if args.multi_modal:
            if args.mm_fusion_mthd == 'concat':
                if modals == 'avl':
                    textf = torch.cat([acouf, visuf, textf1, textf2, textf3, textf4], dim=-1)
                elif modals == 'av':
                    textf = torch.cat([acouf, visuf], dim=-1)
                elif modals == 'vl':
                    textf = torch.cat([visuf, textf1, textf2, textf3, textf4], dim=-1)
                elif modals == 'al':
                    textf = torch.cat([acouf, textf1, textf2, textf3, textf4], dim=-1)
                else:
                    raise NotImplementedError
            elif args.mm_fusion_mthd == 'gated':
                textf = textf
        else:
            if modals == 'a':
                textf = acouf
            elif modals == 'v':
                textf = visuf
            elif modals == 'l':
                textf = textf
            else:
                raise NotImplementedError

        lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]
        if args.multi_modal and args.mm_fusion_mthd == 'gated':
            log_prob, loss_center, loss_contrastive, e_i, e_n, e_t, e_l = model(textf, qmask, umask, lengths, acouf,
                                                                                visuf)
        elif args.multi_modal and args.mm_fusion_mthd == 'concat_subsequently':
            log_prob, loss_center, loss_contrastive, e_i, e_n, e_t, e_l = model([textf1, textf2, textf3, textf4], qmask,
                                                                                umask, lengths, acouf, visuf, epoch)
        elif args.multi_modal and args.mm_fusion_mthd == 'concat_DHT':
            log_prob, e_i, e_n, e_t, e_l, aware_features, spk_emb_vector, data, final_features = model([textf1, textf2, textf3, textf4], qmask,
                                                                                umask, lengths, acouf, visuf, epoch)
        else:
            log_prob, loss_center, loss_contrastive, e_i, e_n, e_t, e_l = model(textf, qmask, umask, lengths)
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        # 特征提取模式跳过损失计算
        if extract_features:
            loss = torch.tensor(0.0).to(log_prob.device)
            loss_center = torch.tensor(0.0).to(log_prob.device)
            loss_contrastive = torch.tensor(0.0).to(log_prob.device)

            # 调试信息
            if final_features is None:
                print(f"⚠️ final_features is None in batch")
            if aware_features is None:
                print(f"⚠️ aware_features is None in batch")
            with torch.no_grad():
                # 保存特征（确保变量不为None）
                if final_features is not None:
                    all_final_features.append(final_features.cpu().detach().numpy())
                    print(f"✅ Saved final features: {final_features.shape}")
                if aware_features is not None:
                    all_aware_features.append(aware_features.cpu().detach().numpy())
                    print(f"✅ Saved aware features: {aware_features.shape}")
        else:
            # loss = loss_function(log_prob, label)
            loss = label_smoothing_loss(log_prob, label, smoothing=0.1)
            if not train and extract_features:
            # 只在评估模式且需要提取特征时保存特征
                all_final_features.append(final_features.cpu().numpy())
                all_aware_features.append(aware_features.cpu().numpy())
            # 计算中心损失
            loss_center, sample_similarities = Multi_Center_Loss(aware_features, label)
            # # 计算对比损失
            loss_contrastive = get_fine_supcontrastive_loss(cuda, aware_features, label, lengths, spk_emb_vector)
            # loss = loss + loss_contrastive
            # loss = loss + loss_center
            # loss = loss + loss_center + loss_contrastive
            # loss = loss / accumulation_steps
            # 引入 alpha 和 beta 权重
            alpha = args.alpha_w
            beta = args.beta_w

            # 总损失
            loss = loss + alpha * loss_center + beta * loss_contrastive

        # 打印各损失组成
        # print(
        #     f"[Epoch {epoch}] CE Loss: {loss.item():.4f}, Center Loss: {loss_center.item():.4f}, Contrastive Loss: {loss_contrastive.item():.4f}")
        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

            # accumulation_counter += 1
            # if accumulation_counter % accumulation_steps == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), float('nan'), float('nan'), [], [], [], [], np.array([]), np.array([])



    # vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    # vids = np.array(vids)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)
    avg_micro_fscore = round(f1_score(labels, preds, average='micro', labels=list(range(1, 7))) * 100, 2)
    avg_macro_fscore = round(f1_score(labels, preds, average='macro') * 100, 2)

    # 合并特征（如果提取了特征）
    final_features_np = np.concatenate(all_final_features, axis=0) if all_final_features else np.array([])
    aware_features_np = np.concatenate(all_aware_features, axis=0) if all_aware_features else np.array([])

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, avg_micro_fscore, avg_macro_fscore, ei, et, en, el, final_features_np, aware_features_np


if __name__ == '__main__':
    # 创建必要的目录
    os.makedirs("./saved_models", exist_ok=True)
    os.makedirs("./saved_features", exist_ok=True)
    os.makedirs("./saved/IEMOCAP", exist_ok=True)

    print("📁 Created necessary directories:")
    print("   - ./saved_models/")
    print("   - ./saved_features/")
    print("   - ./saved/IEMOCAP/")
    path = './saved/IEMOCAP/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')

    parser.add_argument('--graph-model', action='store_true', default=True,
                        help='whether to use graph model after recurrent encoding')

    parser.add_argument('--nodal-attention', action='store_true', default=True,
                        help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')

    parser.add_argument('--windowp', type=int, default=10,
                        help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=10,
                        help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')

    parser.add_argument('--l2', type=float, default=0.00003, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')

    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')

    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')

    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')

    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--graph_type', default='relation', help='relation/GCN3/DeepGCN/MMGCN/MMGCN2')

    parser.add_argument('--use_topic', action='store_true', default=False, help='whether to use topic information')

    parser.add_argument('--alpha', type=float, default=0.2, help='alpha')

    parser.add_argument('--multiheads', type=int, default=6, help='multiheads')

    parser.add_argument('--graph_construct', default='full', help='single/window/fc for MMGCN2; direct/full for others')

    parser.add_argument('--use_gcn', action='store_true', default=False,
                        help='whether to combine spectral and none-spectral methods or not')

    parser.add_argument('--use_residue', action='store_true', default=False,
                        help='whether to use residue information or not')

    parser.add_argument('--multi_modal', action='store_true', default=False,
                        help='whether to use multimodal information')

    parser.add_argument('--mm_fusion_mthd', default='concat',
                        help='method to use multimodal information: concat, gated, concat_subsequently')

    parser.add_argument('--modals', default='avl', help='modals to fusion')

    parser.add_argument('--av_using_lstm', action='store_true', default=False,
                        help='whether to use lstm in acoustic and visual modality')

    parser.add_argument('--Deep_GCN_nlayers', type=int, default=4, help='Deep_GCN_nlayers')

    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')

    parser.add_argument('--use_speaker', action='store_true', default=True, help='whether to use speaker embedding')

    parser.add_argument('--use_modal', action='store_true', default=False, help='whether to use modal embedding')

    parser.add_argument('--norm', default='LN2', help='NORM type')

    parser.add_argument('--testing', action='store_true', default=False, help='testing')

    parser.add_argument('--num_L', type=int, default=3, help='num_hyperconvs')

    parser.add_argument('--num_K', type=int, default=4, help='num_convs')

    parser.add_argument("--device", type=str, default="cuda", help="Computing device.")

    parser.add_argument('--alpha_w', type=float, default=0.1, help='weight for anchor center loss')
    parser.add_argument('--beta_w', type=float, default=0.1, help='weight for SupConLoss contrastive loss')
    parser.add_argument('--warmup', type=int, default=5, help='Number of warmup epochs for auxiliary loss')

    # 在 argument parser 中添加新参数
    parser.add_argument('--extract-features', action='store_true', default=False,
                        help='Extract features after training or for a trained model')
    parser.add_argument('--model-to-extract', type=str, default='',
                        help='Path to model checkpoint for feature extraction')




    args = parser.parse_args()
    today = datetime.datetime.now()
    print(args)
    # 在训练开始前添加清理代码
    if args.Dataset == 'MELD' and os.path.exists('./saved_models/'):
        print("🧹 Cleaning up old model files for MELD...")
        for file in os.listdir('./saved_models/'):
            if file.endswith('.pth.tar'):
                file_path = os.path.join('./saved_models/', file)
                try:
                    os.remove(file_path)
                    print(f"   Removed: {file}")
                except Exception as e:
                    print(f"   Could not remove {file}: {e}")

    # 添加调试信息
    # print(f"🎯 Training for dataset: {args.Dataset}")
    # print(f"📊 Dataset dimensions - Audio: {D_audio}, Visual: {D_visual}, Text: {D_text}")
    # print(f"👥 Number of speakers: {n_speakers}")
    # print(f"🎭 Number of classes: {n_classes}")
    if args.av_using_lstm:
        name_ = args.mm_fusion_mthd + '_' + args.modals + '_' + args.graph_type + '_' + args.graph_construct + 'using_lstm_' + args.Dataset
    else:
        name_ = args.mm_fusion_mthd + '_' + args.modals + '_' + args.graph_type + '_' + args.graph_construct + str(
            args.Deep_GCN_nlayers) + '_' + args.Dataset

    if args.use_speaker:
        name_ = name_ + '_speaker'
    if args.use_modal:
        name_ = name_ + '_modal'

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
    modals = args.modals
    feat2dim = {'IS10': 1582, '3DCNN': 512, 'textCNN': 100, 'bert': 768, 'denseface': 342, 'MELD_text': 600,
                'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.Dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024  # feat2dim['textCNN'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_text']

    if args.multi_modal:
        if args.mm_fusion_mthd == 'concat':
            if modals == 'avl':
                D_m = D_audio + D_visual + D_text
            elif modals == 'av':
                D_m = D_audio + D_visual
            elif modals == 'al':
                D_m = D_audio + D_text
            elif modals == 'vl':
                D_m = D_visual + D_text
            else:
                raise NotImplementedError
        else:
            D_m = 1024
    else:
        if modals == 'a':
            D_m = D_audio
        elif modals == 'v':
            D_m = D_visual
        elif modals == 'l':
            D_m = D_text
        else:
            raise NotImplementedError
    D_g = 512 if args.Dataset == 'IEMOCAP' else 1024
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 512
    n_speakers = 9 if args.Dataset == 'MELD' else 2
    n_classes = 7 if args.Dataset == 'MELD' else 6 if args.Dataset == 'IEMOCAP' else 1

    if args.graph_model:
        seed_everything()

        model = Model(args.base_model,
                      D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                      n_speakers=n_speakers,
                      max_seq_len=200,
                      window_past=args.windowp,
                      window_future=args.windowf,
                      device=args.device,
                      n_classes=n_classes,
                      listener_state=args.active_listener,
                      context_attention=args.attention,
                      dropout=args.dropout,
                      nodal_attention=args.nodal_attention,
                      no_cuda=args.no_cuda,
                      graph_type=args.graph_type,
                      use_topic=args.use_topic,
                      alpha=args.alpha,
                      multiheads=args.multiheads,
                      graph_construct=args.graph_construct,
                      use_GCN=args.use_gcn,
                      use_residue=args.use_residue,
                      D_m_v=D_visual,
                      D_m_a=D_audio,
                      modals=args.modals,
                      att_type=args.mm_fusion_mthd,
                      av_using_lstm=args.av_using_lstm,
                      Deep_GCN_nlayers=args.Deep_GCN_nlayers,
                      dataset=args.Dataset,
                      use_speaker=args.use_speaker,
                      use_modal=args.use_modal,
                      norm=args.norm,
                      num_L=args.num_L,
                      num_K=args.num_K)

        print('Graph NN with', args.base_model, 'as base model.')
        name = 'Graph'

    else:
        if args.base_model == 'GRU':
            model = GRUModel(D_m, D_e, D_h,
                             n_classes=n_classes,
                             dropout=args.dropout)

            print('Basic GRU Model.')


        elif args.base_model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h,
                              n_classes=n_classes,
                              dropout=args.dropout)

            print('Basic LSTM Model.')

        else:
            print('Base model must be one of DialogRNN/LSTM/GRU/Transformer')
            raise NotImplementedError

        name = 'Base'

    if cuda:
        model.cuda()

    if args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1 / 0.086747,
                                          1 / 0.144406,
                                          1 / 0.227883,
                                          1 / 0.160585,
                                          1 / 0.127711,
                                          1 / 0.252668])
        D_audio = 1582  # IEMOCAP音频特征维度
        D_visual = 342  # IEMOCAP视觉特征维度
        D_text = 1024  # IEMOCAP文本特征维度
        n_speakers = 2  # IEMOCAP说话者数量
        n_classes = 6  # IEMOCAP情感类别数量

    if args.Dataset == 'MELD':
        loss_function = FocalLoss()
        D_audio = 300  # MELD音频特征维度
        D_visual = 342  # MELD视觉特征维度
        D_text = 600  # MELD文本特征维度
        n_speakers = 9  # MELD说话者数量
        n_classes = 7  # MELD情感类别数量
    else:
        if args.class_weight:
            if args.graph_model:
                # loss_function = FocalLoss()
                loss_function = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
            else:
                loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            if args.graph_model:
                loss_function = nn.NLLLoss()
            else:
                loss_function = MaskedNLLLoss()

    M=0.1
    feature_dim = 1024
    num_view = 3
    Multi_Center_Loss = My_anchor_Loss(num_view, n_classes, feature_dim, M)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    lr = args.lr

    if args.Dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                   batch_size=batch_size,
                                                                   num_workers=2)
    elif args.Dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=2)
    else:
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_micro_fscore, all_macro_fscore, all_acc, all_loss = [], [], [], [], []

    if args.testing:
        state = torch.load("best_model.pth.tar")
        model.load_state_dict(state)
        print('testing loaded model')
        test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _ = train_or_eval_graph_model(model,
                                                                                                           loss_function,
                                                                                                           Multi_Center_Loss,
                                                                                                           test_loader,
                                                                                                           0, cuda,
                                                                                                           args.modals,
                                                                                                           dataset=args.Dataset)
        print('test_acc:', test_acc, 'test_fscore:', test_fscore)

    early_stopper = EarlyStopping(patience=15)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    for e in range(n_epochs):
        start_time = time.time()

        if args.graph_model:
            train_loss, train_acc, _, _, train_fscore, train_avg_micro_fscore, train_avg_macro_fscore, *_, = train_or_eval_graph_model(
                model, loss_function, Multi_Center_Loss, train_loader, e, cuda, args.modals, optimizer, True, dataset=args.Dataset)
            valid_loss, valid_acc, _, _, valid_fscore, valid_avg_micro_fscore, valid_avg_macro_fscore, *_, = train_or_eval_graph_model(
                model, loss_function, Multi_Center_Loss, valid_loader, e, cuda, args.modals, dataset=args.Dataset)
            test_loss, test_acc, test_label, test_pred, test_fscore, test_avg_micro_fscore, test_avg_macro_fscore, *_, = train_or_eval_graph_model(
                model, loss_function, Multi_Center_Loss, test_loader, e, cuda, args.modals, dataset=args.Dataset)
            all_fscore.append(test_fscore)
            all_micro_fscore.append(test_avg_micro_fscore)
            all_macro_fscore.append(test_avg_macro_fscore)
            all_acc.append(test_acc)

            scheduler.step(test_fscore)

            # 在每个epoch都保存一次checkpoint（用于调试）
            if e % 10 == 0:  # 每10个epoch保存一次
                checkpoint_path = f'./saved_models/checkpoint_epoch{e}_fscore{test_fscore:.2f}.pth.tar'
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': test_loss,
                    'fscore': test_fscore,
                    'args': vars(args),
                    'dataset': args.Dataset  # 明确保存数据集信息
                }, checkpoint_path)
                print(f"💾 Intermediate checkpoint saved: {os.path.basename(checkpoint_path)}")

            early_stopper(test_fscore, model)
            if early_stopper.early_stop:
                print("🛑 Early stopping triggered. Restoring best model...")
                model.load_state_dict(early_stopper.best_model)
                # 创建目录（如果不存在）
                os.makedirs("./saved_models", exist_ok=True)
                # 保存最终的最佳模型
                final_checkpoint = {
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'best_score': early_stopper.best_score,
                    'args': vars(args)
                }
                torch.save(final_checkpoint, './saved_models/final_best_model.pth.tar')
                print("💾 Final best model saved as './saved_models/final_best_model.pth.tar'")
                break

            # 训练结束后如果没有早停，也保存模型
            if not early_stopper.early_stop:
                # 创建目录（如果不存在）
                os.makedirs("./saved_models", exist_ok=True)
                final_checkpoint = {
                    'epoch': n_epochs,
                    'model_state_dict': model.state_dict(),
                    'best_score': max(all_fscore) if all_fscore else 0,
                    'args': vars(args),
                    'dataset': args.Dataset  # 添加数据集信息
                }
                torch.save(final_checkpoint, './saved_models/final_model.pth.tar')
                print("💾 Final model saved as './saved_models/final_model.pth.tar'")

        else:
            train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function, train_loader, e,
                                                                                  optimizer, True)
            valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model, loss_function, valid_loader, e)
            test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model,
                                                                                                                 loss_function,
                                                                                                                 test_loader,
                                                                                                                 e)
            all_fscore.append(test_fscore)

        if best_loss == None or best_loss > test_loss:
            best_loss, best_label, best_pred = test_loss, test_label, test_pred

        if best_fscore == None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label, best_pred = test_label, test_pred
            # test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, test_loader, e, cuda, args.modals, dataset=args.Dataset)

        if args.tensorboard:
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)

        print(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
            format(e + 1, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore,
                   round(time.time() - start_time, 2)))
        if (e + 1) % 10 == 0:
            print('best Weight-F-Score:', max(all_fscore))
            print('best Miceo-F-Score:', max(all_micro_fscore))
            print('best Macro-F-Score:', max(all_macro_fscore))
            print('best Acc-Score:', max(all_acc))
            print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
            print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))

    if args.tensorboard:
        writer.close()
    if not args.testing:
        print('Test performance..')
        print('Wetight-F-Score:', max(all_fscore))
        print('Miceo-F-Score:', max(all_micro_fscore))
        print('Macro-F-Score:', max(all_macro_fscore))
        print('Acc-Score:', max(all_acc))  # 平均准确率
        if not os.path.exists("record_{}_{}_{}.pk".format(today.year, today.month, today.day)):
            with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'wb') as f:
                pk.dump({}, f)
        with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'rb') as f:
            record = pk.load(f)
        key_ = name_
        if record.get(key_, False):
            record[key_].append(max(all_fscore))
        else:
            record[key_] = [max(all_fscore)]
        if record.get(key_ + 'record', False):
            record[key_ + 'record'].append(
                classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
        else:
            record[key_ + 'record'] = [classification_report(best_label, best_pred, sample_weight=best_mask, digits=4)]
        with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'wb') as f:
            pk.dump(record, f)

        print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
        print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))

        # ... 主函数前面的所有代码 ...

    # 特征提取模式 - 这必须放在 main() 函数内部！
    if args.extract_features:
        # 优先使用新训练的模型
        potential_model_paths = [
            './saved_models/final_best_model.pth.tar',
            './saved_models/final_model.pth.tar',
            args.model_to_extract if args.model_to_extract else ''
        ]

        model_path = None
        for path in potential_model_paths:
            if path and os.path.exists(path):
                model_path = path
                print(f"✅ Using model: {os.path.basename(path)}")
                break

        if not model_path:
            print("❌ No trained model found and no model specified!")
            # 列出可用的模型文件
            model_dir = './saved_models/'
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth.tar')]
                if model_files:
                    print("📂 Available model files:")
                    for file in model_files:
                        print(f"   - {file}")
                    # 使用第一个找到的模型文件
                    model_path = os.path.join(model_dir, model_files[0])
                    print(f"💡 Using the first available model: {model_files[0]}")
                else:
                    print("❌ No model files found in ./saved_models/")
                    # return
            else:
                print("❌ saved_models directory does not exist!")
                # return

        print("🚀 Starting feature extraction...")

        # 提取模型特征
        final_feats, aware_feats, labels = extract_model_features(
            model, test_loader, cuda, args.modals,
            model_path, args.Dataset, "proposed"
        )

        if final_feats is not None:
            print("✅ Feature extraction completed!")
            print(f"   Final features: {final_feats.shape}")
            print(f"   Aware features: {aware_feats.shape if aware_feats is not None else 'N/A'}")
            print(f"   Labels: {labels.shape if labels is not None else 'N/A'}")

            # 保存特征
            os.makedirs("./saved_features", exist_ok=True)
            if final_feats is not None and len(final_feats) > 0:
                np.save(f'./saved_features/proposed_{args.Dataset}_final_features.npy', final_feats)
                print(f"💾 Final features saved")
            if aware_feats is not None and len(aware_feats) > 0:
                np.save(f'./saved_features/proposed_{args.Dataset}_aware_features.npy', aware_feats)
                print(f"💾 Aware features saved")
            if labels is not None and len(labels) > 0:
                np.save(f'./saved_features/proposed_{args.Dataset}_labels.npy', labels)
                print(f"💾 Labels saved")
        else:
            print("❌ Feature extraction failed!")