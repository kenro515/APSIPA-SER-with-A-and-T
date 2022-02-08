import os
import yaml
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as data

from transformers import BertForSequenceClassification
from layers.CNN_BiLSTM_Attention import CNN_BiLSTM_Attention
from layers.Fusion import EarlyConcat_LateLinear, EarlyMul_LateLinear, EarlyConcat_LateFC, EarlyMul_LateFC

from dataloader_transformers import MyDataset_audio_and_text
from miscellaneous.plot_results import plot_cm, plot_curve
from miscellaneous.schedule import WarmupConstantSchedule

torch.manual_seed(1234)
np.random.seed(1234)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True


def split_audio(p_input_audio=None):
    p_split_inputs = []

    split_pvot = (p_input_audio.shape[3] // (100 // 2)) - 1
    for i in range(split_pvot):
        seg_input = p_input_audio[
            :, :, :, ((100 // 2) * i): (100 + ((100 // 2) * i))]
        p_split_inputs.append(seg_input)

    return p_split_inputs


def train(net_audio, net_text, net_cat, Latefusion_flag, train_loader, optimizer, criterion):
    net_cat.train()
    epoch_loss, epoch_corrects, total = 0.0, 0.0, 0.0

    for (input_audio, input_ids, attention_mask, labels) in tqdm(train_loader):
        input_audio = input_audio.to(device)
        text_dim = input_ids.shape[2]
        input_ids = input_ids.view(-1, text_dim).to(device)
        attention_mask = attention_mask.view(-1, text_dim).to(device)
        labels = labels.to(device)
        target = torch.max(labels, 1)[1]

        # ========== (1) Text Emotion Recognition ==========

        outputs = net_text(
            input_ids,
            token_type_ids=None,
            attention_mask=attention_mask,
            labels=target
        )
        logits_text = outputs.logits
        h_bert = outputs.hidden_states

        # ========== (2) Speech Emotion Recognition ==========

        split_inputs = []
        split_inputs = split_audio(input_audio)

        logits_audio, audio_feature = net_audio(split_inputs)

        # ========== (3) Fusion Emotion Recognition ==========

        if Latefusion_flag == 0:
            logits_fusion = net_cat(
                h_bert[-1].sum(dim=1), audio_feature.sum(dim=1))
            pred_fusion = logits_text + logits_audio + logits_fusion
        else:
            pred_fusion = net_cat(
                h_bert[-1].sum(dim=1),
                audio_feature.sum(dim=1),
                logits_text,
                logits_audio
            )
        loss = criterion(pred_fusion, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predict = torch.max(pred_fusion, 1)

        epoch_loss += loss.item()
        epoch_corrects += torch.sum(predict == target).item()
        total += labels.size(0)

    train_loss = epoch_loss / len(train_loader)
    train_acc = epoch_corrects / total

    return net_cat, train_loss, train_acc


def valid(net_audio, net_text, net_cat, Latefusion_flag, val_loader, criterion):
    net_audio.eval()
    net_text.eval()
    net_cat.eval()

    epoch_loss, epoch_corrects, total = 0.0, 0.0, 0.0

    target_lists = torch.zeros(0, dtype=torch.long, device='cpu')
    predict_lists = torch.zeros(0, dtype=torch.long, device='cpu')

    with torch.no_grad():
        for (input_audio, input_ids, attention_mask, labels) in tqdm(val_loader):
            input_audio = input_audio.to(device)
            text_dim = input_ids.shape[2]
            input_ids = input_ids.view(-1, text_dim).to(device)
            attention_mask = attention_mask.view(-1, text_dim).to(device)
            labels = labels.to(device)
            target = torch.max(labels, 1)[1]

            # ========== (1) Text Emotion Recognition ==========

            outputs = net_text(
                input_ids,
                token_type_ids=None,
                attention_mask=attention_mask,
                labels=target
            )
            logits_text = outputs.logits
            h_bert = outputs.hidden_states

            # ========== (2) Speech Emotion Recognition ==========

            split_inputs = []
            split_inputs = split_audio(input_audio)

            logits_audio, audio_feature = net_audio(split_inputs)

            # ========== (3) Fusion Emotion Recognition ==========

            if Latefusion_flag == 0:
                logits_fusion = net_cat(
                    h_bert[-1].sum(dim=1), audio_feature.sum(dim=1))
                pred_fusion = logits_text + logits_audio + logits_fusion
            else:
                pred_fusion = net_cat(
                    h_bert[-1].sum(dim=1),
                    audio_feature.sum(dim=1),
                    logits_text,
                    logits_audio
                )
            loss = criterion(pred_fusion, target)

            _, predict = torch.max(pred_fusion, 1)

            target_lists = torch.cat([target_lists, target.cpu()])
            predict_lists = torch.cat([predict_lists, predict.cpu()])

            epoch_loss += loss.item()
            epoch_corrects += torch.sum(predict == target).item()
            total += labels.size(0)

        val_loss = epoch_loss / len(val_loader)
        val_acc = epoch_corrects / total

    return net_cat, val_loss, val_acc, target_lists.numpy(), predict_lists.numpy()


if __name__ == "__main__":
    # ================== [1] Set up ==================
    # load param
    with open('hyper_param.yaml', 'r') as file:
        config = yaml.safe_load(file.read())

    in_dir_text = config['dataset_setting']['in_dir_text']
    in_dir_mcep = config['dataset_setting']['in_dir_mcep']
    in_dim_mcep = config['dataset_setting']['in_dim_mcep']

    param_dir_text = config['pretrained_model_setting']['in_dir_text']
    param_dir_mcep = config['pretrained_model_setting']['in_dir_mcep']

    model_fusion = config['model_setting']['model_fusion']

    max_len_text = config['dataset_setting']['max_len_text']
    max_len_audio = config['dataset_setting']['max_len_audio']

    epochs = config['training_setting']['epoch']
    batch_size = config['training_setting']['batch_size']
    d_model = config['training_setting']['d_model']
    learning_rate = config['training_setting']['learning_rate']
    warmup_rate = config['training_setting']['warmup_rate']
    early_stopping = config['training_setting']['early_stopping']

    Latefusion_flag = 0

    fold_idx = 0
    fold_num = 5
    cross_validation = 0.0
    cv_lists = []

    time_now = datetime.datetime.now()
    os.makedirs(
        "./results/{}/accuracy_curve".format(str(time_now.date())), exist_ok=True)
    os.makedirs(
        "./results/{}/confusion_matrix".format(str(time_now.date())), exist_ok=True)
    os.makedirs(
        "./results/{}/learning_curve".format(str(time_now.date())), exist_ok=True)
    os.makedirs(
        "./results/{}/model_param".format(str(time_now.date())), exist_ok=True)

    # ================== [2] Training and Validation ==================
    print("Start training!")
    for fold_idx in range(fold_num):
        net_audio = CNN_BiLSTM_Attention(d_model=d_model)

        net_text = BertForSequenceClassification.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking",
            num_labels=4,
            output_attentions=False,
            output_hidden_states=True
        )

        if "EarlyConcat_LateLinear" in model_fusion:
            net_cat = EarlyConcat_LateLinear()
            Latefusion_flag = 0
        elif "EarlyMul_LateLinear" in model_fusion:
            net_cat = EarlyMul_LateLinear()
            Latefusion_flag = 0
        elif "EarlyConcat_LateFC" in model_fusion:
            net_cat = EarlyConcat_LateFC()
            Latefusion_flag = 1
        elif "EarlyMul_LateFC" in model_fusion:
            net_cat = EarlyMul_LateFC()
            Latefusion_flag = 1
        else:
            raise Exception(
                'Please select the correct fusion model and retry this code...')

        # ==================================
        net_text.load_state_dict(
            torch.load(
                "{}/model_param/SER_JTES_TEXT_fold{}_Param.pth".format(
                    param_dir_text, fold_idx),
                map_location=device)
        )
        net_audio.load_state_dict(
            torch.load(
                "{}/model_param/SER_JTES_fold{}_Param.pth".format(
                    param_dir_mcep, fold_idx+1),
                map_location=device)
        )

        # Freezing
        for i, param in enumerate(net_audio.parameters()):
            param.requires_grad = False
        for i, param in enumerate(net_text.parameters()):
            param.requires_grad = False
        # ==================================

        if fold_idx == 0:
            print(net_audio)
            print(net_text)
            print(net_cat)

        net_audio.to(device)
        net_text.to(device)
        net_cat.to(device)

        optimizer = torch.optim.Adam(
            net_cat.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        criterion.to(device)
        scheduler = WarmupConstantSchedule(
            optimizer, warmup_epochs=epochs * warmup_rate)

        train_acc_curve = []
        valid_acc_curve = []
        train_loss_curve = []
        valid_loss_curve = []

        train_dataset = MyDataset_audio_and_text(
            in_path_mcep="{}/train/**/**/*.npy".format(in_dir_mcep),
            in_dir_text=in_dir_text,
            in_dim_mcep=in_dim_mcep,
            max_len_audio=max_len_audio,
            max_len_text=max_len_text,
            fold=fold_idx
        )

        valid_dataset = MyDataset_audio_and_text(
            in_path_mcep="{}/test/**/**/*.npy".format(in_dir_mcep),
            in_dir_text=in_dir_text,
            in_dim_mcep=in_dim_mcep,
            max_len_audio=max_len_audio,
            max_len_text=max_len_text,
            fold=fold_idx
        )

        train_loader = data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size
        )
        valid_loader = data.DataLoader(
            valid_dataset,
            shuffle=False,
            batch_size=batch_size
        )

        print("train_loader:{}".format(len(train_loader)))
        print("valid_loader:{}".format(len(valid_loader)))

        patience = 0
        for epoch in tqdm(range(epochs)):
            net_cat, train_loss, train_acc = train(
                net_audio, net_text, net_cat, Latefusion_flag, train_loader, optimizer, criterion)
            net_cat, val_loss, val_acc, target_list, predict_list = valid(
                net_audio, net_text, net_cat, Latefusion_flag, valid_loader, criterion)

            print('epoch {}/{} | train_loss {:.8f} train_acc {:.8f} | valid_loss {:.8f} valid_acc {:.8f}'.format(
                epoch+1, epochs, train_loss, train_acc, val_loss, val_acc
            ))

            scheduler.step()

            # ===== early-stopling =====
            if train_loss < val_loss:
                patience += 1
                if patience > 1:
                    break
            else:
                patience = 0
            # ==========================

            train_loss_curve.append(train_loss)
            train_acc_curve.append(train_acc)
            valid_loss_curve.append(val_loss)
            valid_acc_curve.append(val_acc)

        torch.save(
            net_cat.state_dict(),
            "./results/{}/model_param/SER_JTES_fold{}_Param.pth".format(
                str(time_now.date()), fold_idx)
        )
        cv_lists.append(val_acc)
        cross_validation += val_acc

        plot_curve(
            train_acc_curve,
            valid_acc_curve,
            x_label=config['plot_acc_curve_setting']['acc_curve_x_label'],
            y_label=config['plot_acc_curve_setting']['acc_curve_y_label'],
            title=config['plot_acc_curve_setting']['acc_curve_title'],
            fold_idx=fold_idx,
            dir_path_name=str(time_now.date())
        )
        plot_curve(
            train_loss_curve,
            valid_loss_curve,
            x_label=config['plot_loss_curve_setting']['loss_curve_x_label'],
            y_label=config['plot_loss_curve_setting']['loss_curve_y_label'],
            title=config['plot_loss_curve_setting']['loss_curve_title'],
            fold_idx=fold_idx,
            dir_path_name=str(time_now.date())
        )
        plot_cm(
            target_list,
            predict_list.T,
            x_label=config['plot_cm_setting']['cm_x_label'],
            y_label=config['plot_cm_setting']['cm_y_label'],
            dir_path_name=str(time_now.date())
        )

    # ================== [3] Plot results of CV ==================
    print("cross validation:{}".format(cv_lists))
    print("cross validation [ave]:{}".format(cross_validation / fold_num))
    print("Finished!")
