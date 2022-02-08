import glob
import random
import torch
import torch.utils.data as data
import numpy as np

from transformers import BertJapaneseTokenizer
from torchvision import transforms

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

trans_t = BertJapaneseTokenizer.from_pretrained(
    'cl-tohoku/bert-base-japanese-whole-word-masking')


def prepare_leave_one_group_out_fusion(p_paths, p_mode, p_fold):
    spk_idx_list = [
        ["m01", "m11", "m21", "m31", "m41", "f01", "f11", "f21", "f31", "f41",
            "m02", "m12", "m22", "m32", "m42", "f02", "f12", "f22", "f32", "f42"],
        ["m03", "m13", "m23", "m33", "m43", "f03", "f13", "f23", "f33", "f43",
            "m04", "m14", "m24", "m34", "m44", "f04", "f14", "f24", "f34", "f44"],
        ["m05", "m15", "m25", "m35", "m45", "f05", "f15", "f25", "f35", "f45",
            "m06", "m16", "m26", "m36", "m46", "f06", "f16", "f26", "f36", "f46"],
        ["m07", "m17", "m27", "m37", "m47", "f07", "f17", "f27", "f37", "f47",
            "m08", "m18", "m28", "m38", "m48", "f08", "f18", "f28", "f38", "f48"],
        ["m09", "m19", "m29", "m39", "m49", "f09", "f19", "f29", "f39", "f49",
            "m10", "m20", "m30", "m40", "m50", "f10", "f20", "f30", "f40", "f50"]
    ]

    fold_idx_train = [
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [0, 2, 3, 4],
        [0, 1, 3, 4],
        [0, 1, 2, 4]
    ]

    fold_idx_test = [
        4, 0, 1, 2, 3
    ]

    seg_fold_list_out = []

    if 'train' in p_mode:
        print("train path adjusting...")
        print("fold_idx_train:{}".format(fold_idx_train[p_fold]))

        for p_path in p_paths:
            for fold_idx in fold_idx_train[p_fold]:
                for spk_idx in spk_idx_list[fold_idx]:
                    if spk_idx in p_path:
                        seg_fold_list_out.append(p_path)

        print(len(seg_fold_list_out))
    else:
        print("test path adjusting...")
        print("fold_idx_test:{}".format(fold_idx_test[p_fold]))

        for p_path in p_paths:
            for spk_idx in spk_idx_list[fold_idx_test[p_fold]]:
                if spk_idx in p_path:
                    seg_fold_list_out.append(p_path)

        print(len(seg_fold_list_out))

    return seg_fold_list_out


class MyDataset_audio_and_text(data.Dataset):
    def __init__(self, in_path_mcep=None, in_dir_text=None, in_dim_mcep=None, max_len_audio=None, max_len_text=None, fold=None):
        super(MyDataset_audio_and_text, self).__init__()

        # ** check! ** 'p_mode' is train or test
        self.mcep_paths = prepare_leave_one_group_out_fusion(
            sorted(glob.glob(in_path_mcep)),
            p_mode=in_path_mcep.split('/')[5],
            p_fold=fold
        )
        self.data_len = len(self.mcep_paths)

        self.in_dim_mcep = in_dim_mcep
        self.max_audio_len = max_len_audio
        self.max_text_len = max_len_text
        self.in_dir_text = in_dir_text

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        p_audio = self.mcep_paths[index]

        p_filename = p_audio.split('/')[-1].split('.')[0]

        mode = 'kanji/' + p_audio.split('/')[6]
        p_text = \
            self.in_dir_text + '/' + mode + '/' + \
            p_filename.split('_')[1] + '_' + p_filename.split('_')[2] + '.txt'

        # ===== processing audio data =====
        audio_data = np.load(p_audio)

        # triming
        if audio_data.shape[0] > self.max_audio_len:
            audio_data = audio_data[0:self.max_audio_len, :]
        else:
            # padding
            padding_size = self.max_audio_len-audio_data.shape[0]
            buff = np.zeros([padding_size, self.in_dim_mcep])

            audio_data = np.concatenate([audio_data, buff])

        audio_data = np.transpose(audio_data, (1, 0))
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        audio_data = transform(audio_data)

        # ** check! ** make categorical emotion label
        emo_list = p_audio.split('/')[6]
        emo_label = np.zeros(4, dtype='int64')
        if emo_list == "ang":
            emo_label[0] = 1
        elif emo_list == "joy":
            emo_label[1] = 1
        elif emo_list == "sad":
            emo_label[2] = 1
        else:
            emo_label[3] = 1

        # ===== processing text data =====
        with open(p_text) as f:
            row = f.read()
            encoded_dict = trans_t.encode_plus(
                row,
                add_special_tokens=True,
                padding='max_length',
                max_length=self.max_text_len,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

        return audio_data.type(torch.cuda.FloatTensor), \
            encoded_dict['input_ids'], \
            encoded_dict['attention_mask'], \
            torch.from_numpy(emo_label)
