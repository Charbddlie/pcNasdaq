import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import torch
import os
from utils import MAE
from torch.utils.data import DataLoader, Dataset

def get_seq_dataset(dataset, batch_size, seq_len, verbose):
    class rnn_eval():
        def __init__(self, batch_size):
            self.batch_size = batch_size
            self.test = False
            self.no_improve_cnt = 0

        def reset(self, epoch_num):
            raise NotImplementedError("Subclasses should implement this method.")

        def train_reset(self, epoch_num):
            self.reset(epoch_num)
            self.test = False
            self.no_improve_cnt = 0

        def test_reset(self, epoch_num):
            self.reset(epoch_num)
            self.test = True
            self.no_improve_cnt = 0

        def summarize(self, p=print):
            raise NotImplementedError("Subclasses should implement this method.")

        def __call__(self, y, t):
            raise NotImplementedError("Subclasses should implement this method.")
        
    class acc_eval(rnn_eval):
        def __init__(self, batch_size):
            super().__init__(batch_size)
            self.reset(0)
            self.max_acc = 0
            self.max_acc_epoch = -1

        def reset(self, epoch_num):
            self.count = 0
            self.correct = 0
            self.epoch_num = epoch_num

        def summarize(self, p=print):
            accuracy = self.correct / self.count if self.count != 0 else 0
            p(f"Total Accuracy: {accuracy}")
            self.no_improve_cnt += 1
            if self.test and self.max_acc < accuracy:
                self.max_acc = accuracy
                self.max_acc_epoch = self.epoch_num
                self.no_improve_cnt = 0
            return self.max_acc, self.max_acc_epoch, self.no_improve_cnt

        def __call__(self, y, t):
            accuracy = 0
            L, _, B = t.shape
            for i in range(len(y)): # this loop is over the seq_len
                for b in range(B):
                    if torch.argmax(t[i,:,b]) == torch.argmax(y[i][:,b]):
                        accuracy+=1
            if verbose: print(f"Batch Accuracy: {accuracy / (L * B) if L * B != 0 else 0}")
            self.correct += accuracy
            self.count += L * B
        
    class loss_eval(rnn_eval):
        def __init__(self, batch_size):
            super().__init__(batch_size)
            self.reset(0)
            self.min_mse_loss = 1e1000
            self.min_mse_loss_epoch = -1
            self.min_mae_loss = 1e1000
            self.min_mae_loss_epoch = -1

        def reset(self, epoch_num):
            self.total_mse_loss = 0
            self.total_mae_loss = 0
            self.batch_cnt = 0
            self.epoch_num = epoch_num
        
        def summarize(self, p=print):
            avg_mse_loss = self.total_mse_loss / self.batch_cnt
            avg_mae_loss = self.total_mae_loss / self.batch_cnt
            p(f"Total MSE Loss: {avg_mse_loss}\tTotal MAE Loss: {avg_mae_loss}")
            self.no_improve_cnt += 1
            if self.test and self.min_mse_loss > avg_mse_loss:
                self.min_mse_loss = avg_mse_loss
                self.min_mse_loss_epoch = self.epoch_num
                self.no_improve_cnt = 0
            if self.test and self.min_mae_loss > avg_mae_loss:
                self.min_mae_loss = avg_mae_loss
                self.min_mae_loss_epoch = self.epoch_num
                self.no_improve_cnt = 0
            return self.min_mse_loss, self.min_mse_loss_epoch, self.no_improve_cnt

        def __call__(self, output_seq, target_seq):
            mse_loss = F.mse_loss(output_seq, target_seq).item()
            mae_loss = MAE(output_seq, target_seq).item()
            if verbose: print(f"Batch MSE Loss: {mse_loss}\tBatch MAE Loss: {mae_loss}")
            self.total_mse_loss += mse_loss
            self.total_mae_loss += mae_loss
            self.batch_cnt += 1

    rnn_prefix = "rnn_data"
    os.makedirs(rnn_prefix, exist_ok=True)
    if dataset == "nasdaq100_extended":
        url = "https://cseweb.ucsd.edu/~yaq007/nasdaq100.zip"
        dir = os.path.join(rnn_prefix, "nasdaq100_extended")
        os.makedirs(dir, exist_ok=True)
        path_to_file = os.path.join(dir, 'extended_non_padding.csv')
        if not os.path.exists(path_to_file):
            print("downloading nasdaq100_extended")
            os.system('''
                        if [ ! -f nasdaq100.zip ]; then
                            wget https://cseweb.ucsd.edu/~yaq007/nasdaq100.zip
                        fi
                        mkdir -p rnn_data/nasdaq100_extended
                        cd rnn_data/nasdaq100_extended
                        mv ../../nasdaq100.zip .
                        unzip nasdaq100.zip
                        mv nasdaq100/extended/extended_non_padding.csv .
                        rm -rf nasdaq100
                        rm -rf __MACOSX
                    ''')
        nasdaq_data = pd.read_csv(path_to_file)
        input_size, output_size = 1, 1
        nasdaq_data = nasdaq_data[['CTAS']]
        nasdaq_data['CTAS'][:] = nasdaq_data['CTAS'].interpolate()
        nasdaq_data['CTAS'][:] = (nasdaq_data['CTAS'] - nasdaq_data['CTAS'].mean()) / nasdaq_data['CTAS'].std()
        seqs = []
        for i in range(len(nasdaq_data) - seq_len):
            seq = nasdaq_data['CTAS'].iloc[i:i + seq_len + 1].values
            if(len(seq)!=seq_len+1): continue
            seqs.append(seq)

        tensors = []

        # 先截取seqs的长度为batch_size的整数倍
        seqs = seqs[:len(seqs) - len(seqs) % batch_size]
        
        for seq in seqs:
            input_tensor = torch.zeros((seq_len, 1))
            output_tensor = torch.zeros((seq_len, 1))
            input_tensor[:, 0] = torch.tensor(seq[:seq_len])
            output_tensor[:, 0] = torch.tensor(seq[1:seq_len + 1])
            tensors.append((input_tensor, output_tensor))

        split_ratio = 0.8
        split_index = int(len(tensors) * split_ratio)
        train_tensors = tensors[:split_index]
        test_tensors = tensors[split_index:]

        trainloader = DataLoader(train_tensors, batch_size=batch_size, shuffle=False)
        testloader = DataLoader(test_tensors, batch_size=batch_size, shuffle=False)

        rnn_eval = loss_eval(batch_size)
            
    elif dataset == "trigonometric":
        def generate_random_tensors(seq_len):
            x = torch.rand(seq_len, 1) * (2 * np.pi) - np.pi  # 随机生成seq_len个x，范围在[-π, π]
            # x, _ = torch.sort(x, dim=0)  # 给x排个序
            # x = torch.linspace(-2 * np.pi, 2 * np.pi, seq_len).unsqueeze(1)  # 生成固定的seq_len个x，范围在[-2π, 2π]
            y = -torch.sin(x)
            x = x / np.pi
            return x, y

        train_dataset = []
        test_dataset = []
        input_size, output_size = 1, 1
        for _ in range(batch_size * 50):
            x_tensor, y_tensor = generate_random_tensors(seq_len)
            train_dataset.append((x_tensor, y_tensor))
        for _ in range(batch_size * 10):
            x_tensor, y_tensor = generate_random_tensors(seq_len)
            test_dataset.append((x_tensor, y_tensor))

        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        rnn_eval = loss_eval(batch_size)
    elif dataset == 'stock':
        pass
    else:
        raise Exception("dataset: " + str(dataset) + " not supported")

    print("Setup data:")
    print("Train: ",len(trainloader.dataset))
    print("Test: ", len(testloader.dataset))
    return trainloader, testloader, input_size, output_size, rnn_eval
