import torch
import torchvision.transforms as transforms
from utils import *

class BP_RNN_Model(object):
    def __init__(self, args, device='cuda') -> None:
        self.args = args
        self.device = device
        self.seq_len = args.seq_len
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.fn = args.fn
        self.fn_deriv = args.fn_deriv
        self.weight_learning_rate = args.weight_learning_rate
        self.clamp_val = 50
        #weights
        self.Wh = torch.empty([self.hidden_size, self.hidden_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.Wx = torch.empty([self.hidden_size, self.input_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.Wy = torch.empty([self.output_size, self.hidden_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.h0 = torch.empty([self.hidden_size, self.batch_size]).normal_(mean=0.0,std=0.05).to(self.device)

        self.hu = torch.empty([self.seq_len+1, self.hidden_size, self.batch_size]).to(self.device)
        self.y = torch.empty([self.seq_len, self.output_size, self.batch_size]).to(self.device)

    def forward(self, inputs_seq):
        # L V B
        self.inputs = inputs_seq.clone()
        self.hu[0] = self.h0
        for i, inp in enumerate(inputs_seq):
            self.hu[i+1] = self.fn(self.Wh @ self.hu[i] + self.Wx @ inp)
            self.y[i] = linear(self.Wy @ self.hu[i+1])
        return self.y
    
    def update_weights(self, target_seq):
        dhs = torch.zeros_like(self.hu).to(self.device)
        dys = torch.zeros_like(self.y).to(self.device)
        dWy = torch.zeros_like(self.Wy).to(self.device)
        dWx = torch.zeros_like(self.Wx).to(self.device)
        dWh = torch.zeros_like(self.Wh).to(self.device)

        for i, tar in reversed(list(enumerate(target_seq))):
            dys[i] = tar - self.y[i]
            dh = self.Wy.T @ (dys[i] * linear_deriv(self.Wy @ self.hu[i+1]))
            if i < len(target_seq) - 1:
                fn_deriv =  self.fn_deriv(self.Wh @ self.hu[i+1] + self.Wx @ self.inputs[i+1])
                dh += self.Wh.T @ (dhs[i+1] * fn_deriv)
            dhs[i]= dh
        for i,inp in reversed(list(enumerate(self.inputs))):
            fn_deriv = self.fn_deriv(self.Wh @ self.hu[i] + self.Wx @ inp)
            dWy += (dys[i] * linear_deriv(self.Wy @ self.hu[i+1])) @ self.hu[i+1].T
            dWx += (dhs[i] * fn_deriv) @ inp.T
            dWh += (dhs[i] * fn_deriv) @ self.hu[i].T
        self.Wy += self.weight_learning_rate * torch.clamp(dWy, -self.clamp_val, self.clamp_val)
        self.Wx += self.weight_learning_rate * torch.clamp(dWx, -self.clamp_val, self.clamp_val)
        self.Wh += self.weight_learning_rate * torch.clamp(dWh, -self.clamp_val, self.clamp_val)
  
class PC_RNN_Model(object):
    def __init__(self, args, device='cuda') -> None:
        self.args = args
        self.device = device
        self.seq_len = args.seq_len
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.fn = args.fn
        self.fn_deriv = args.fn_deriv
        self.weight_learning_rate = args.weight_learning_rate
        self.inference_learning_rate = args.inference_learning_rate
        self.inference_steps = args.inference_steps
        self.fixed_predictions = args.fixed_predictions
        self.clamp_val = 50
        #weights
        self.Wh = torch.empty([self.hidden_size, self.hidden_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.Wx = torch.empty([self.hidden_size, self.input_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.Wy = torch.empty([self.output_size, self.hidden_size]).normal_(mean=0.0,std=0.05).to(self.device)
        self.h0 = torch.empty([self.hidden_size, self.batch_size]).normal_(mean=0.0,std=0.05).to(self.device)

        self.hu = torch.empty([self.seq_len+1, self.hidden_size, self.batch_size]).to(self.device)
        self.hx = torch.zeros_like(self.hu)
        self.y = torch.empty([self.seq_len, self.output_size, self.batch_size]).to(self.device)

    def forward(self, inputs_seq):
        # L V B
        self.inputs = inputs_seq.clone()
        self.hu[0] = self.h0.clone()
        self.hx[0] = self.h0.clone()
        for i, inp in enumerate(inputs_seq):
            self.hu[i+1] = self.fn(self.Wh @ self.hu[i] + self.Wx @ inp)
            self.hx[i+1] = self.hu[i+1].clone()
            self.y[i] = linear(self.Wy @ self.hu[i+1])
        return self.y
    
    def update_weights(self, target_seq):
        with torch.no_grad():
            # the last dim of ehs is not used, because we don't need hx[0] - hu[0]
            ehs = torch.zeros_like(self.hu).to(self.device)
            eys = torch.zeros_like(self.y).to(self.device)
            for i, tar in reversed(list(enumerate(target_seq))):
                for n in range(self.inference_steps):
                    # for hx,hu,deltah,ehs, i+1 means time t
                    # for eys,y, i means time t
                    eys[i] = tar - self.y[i]
                    if self.fixed_predictions == False:
                        self.hu[i+1] = self.fn(self.Wh @ self.hx[i] + self.Wx @ self.inputs[i])
                    ehs[i] = self.hx[i+1] - self.hu[i+1] # x-u
                    deltah = -ehs[i].clone()
                    deltah += self.Wy.T @ (eys[i] * linear_deriv(self.Wy @ self.hu[i+1]))
                    if i < len(target_seq)-1:
                        fn_deriv =  self.fn_deriv(self.Wh @ self.hu[i+1] + self.Wx @ self.inputs[i]) # current layer
                        deltah += self.Wh.T @ (ehs[i+1] * fn_deriv)
                    self.hx[i+1] += self.inference_learning_rate * deltah
                    last_deltah = deltah
                    if self.fixed_predictions == False:
                        self.y[i] = linear(self.Wy @ self.hx[i+1])
                    # print(f"iter {n} {self.hx[i+1][0]}")
                    
            dWy = torch.zeros_like(self.Wy).to(self.device)
            dWx = torch.zeros_like(self.Wx).to(self.device)
            dWh = torch.zeros_like(self.Wh).to(self.device)
            for i,inp in reversed(list(enumerate(self.inputs))):
                fn_deriv = self.fn_deriv(self.Wh @ self.hu[i] + self.Wx @ inp)
                dWy += (eys[i] * linear_deriv(self.Wy @ self.hu[i+1])) @ self.hu[i+1].T
                dWx += (ehs[i] * fn_deriv) @ inp.T
                dWh += (ehs[i] * fn_deriv) @ self.hu[i].T
            self.Wy += self.weight_learning_rate * torch.clamp(dWy, -self.clamp_val, self.clamp_val)
            self.Wx += self.weight_learning_rate * torch.clamp(dWx, -self.clamp_val, self.clamp_val)
            self.Wh += self.weight_learning_rate * torch.clamp(dWh, -self.clamp_val, self.clamp_val)

