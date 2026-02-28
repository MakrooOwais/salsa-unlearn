import torch
import torch.nn as nn
from copy import deepcopy
import time
import math

from .unlearn_method import UnlearnMethod

def cycle(dl):
    while True:
        for data in dl:
            yield data

class SALSA(UnlearnMethod):
    def __init__(self, model, loss_function, save_path, args) -> None:
        super().__init__(model, loss_function, save_path, args)
        self.num_classes = args.num_classes
        self.seed = args.seed
        self.eval = True
        
        self.opt = 'adamw'
        self.lr = 9e-4
        self.n_iters = 1500  
        self.log_freq = 500
        
        self.alpha_min = 0.3
        self.alpha_max = 0.6
        self.shift = 5e-3
        self.gamma = 1e-2
        self.slope = 5e-4
        
        self.alpha = self.alpha_min
        
        # Save original model
        self.orig_model = deepcopy(self.model)
        self.orig_model.eval()
        
        self.criterion = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        
        # We need to compute layer-wise lambda
        self.init_params()

    def init_params(self):
        self.layer_wise_lambda_min = []
        self.layer_wise_lambda_max = []
        self.layer_wise_lambda = []
        
        for idx, param in enumerate(self.orig_model.parameters()):
            param.requires_grad = False
            l_min = self.slope * (idx + 1) + self.shift
            self.layer_wise_lambda_min.append(l_min)
            self.layer_wise_lambda_max.append(l_min + self.gamma)
            self.layer_wise_lambda.append(l_min)
            
        self.layer_wise_lambda_min = torch.tensor(self.layer_wise_lambda_min).cuda()
        self.layer_wise_lambda_max = torch.tensor(self.layer_wise_lambda_max).cuda()
        self.layer_wise_lambda = torch.tensor(self.layer_wise_lambda).cuda()
        
        # Flip so input layers have higher lambda
        self.layer_wise_lambda_min = torch.flip(self.layer_wise_lambda_min, dims=(0,))
        self.layer_wise_lambda_max = torch.flip(self.layer_wise_lambda_max, dims=(0,))
        self.layer_wise_lambda = torch.flip(self.layer_wise_lambda, dims=(0,))
        
    def prepare_unlearn(self, unlearn_dataloaders: dict) -> None:
        self.unlearn_dataloaders = unlearn_dataloaders
        
    def create_target(self, curr_model_preds, orig_model_preds):
        target_probs = curr_model_preds.clone()
        mask = torch.zeros_like(target_probs).scatter_(
            1, orig_model_preds.argmax(-1).unsqueeze(1), 1
        )
        reduction = self.alpha * target_probs * mask
        target_probs -= reduction
        if self.num_classes == 2:
            target_probs[~mask.to(torch.bool)] = 1 - target_probs[mask.to(torch.bool)]
        else:
            div = (target_probs * (1 - mask)).sum(keepdim=True, dim=-1)
            target_probs += (target_probs * (1 - mask) * reduction.sum(-1, True)) / div
        return target_probs

    def get_unlearned_model(self) -> nn.Module:
        forget_trainloader = self.unlearn_dataloaders['forget_train']
        forget_train_iter = cycle(forget_trainloader)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
        for step in range(self.n_iters):
            self.model.train()
            
            # Step lambda and alpha schedulers
            lambda_iter = step
            self.layer_wise_lambda = self.layer_wise_lambda_max - 0.5 * (
                self.layer_wise_lambda_max - self.layer_wise_lambda_min
            ) * (1 + torch.cos(torch.tensor((lambda_iter * math.pi) / max(1, self.n_iters - 1))))
            
            alpha_iter = step
            self.alpha = self.alpha_max - 0.5 * (self.alpha_max - self.alpha_min) * (
                1 + torch.cos(torch.tensor((alpha_iter * math.pi) / max(1, self.n_iters - 1)))
            )
            
            x, y = next(forget_train_iter)
            x, y = x.cuda(), y.cuda()
            
            curr_model_preds = torch.nn.Softmax(dim=-1)(self.model(x))
            with torch.no_grad():
                orig_model_probs = torch.nn.Softmax(dim=-1)(self.orig_model(x))
                modified_probs = self.create_target(curr_model_preds, orig_model_probs)
                
            loss_base = self.criterion(
                torch.log(curr_model_preds + 1e-10), torch.log(modified_probs + 1e-10)
            )
            
            param_diff = 0
            for idx, (original_param, current_param) in enumerate(
                zip(self.orig_model.parameters(), self.model.parameters())
            ):
                param_diff += torch.sum(
                    ((current_param - original_param) ** 2) * self.layer_wise_lambda[idx]
                )
                
            loss = loss_base + param_diff
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if self.eval and (step + 1) % self.log_freq == 0:
                print(f"step={step+1} Loss:{loss.item():.4f} Alpha:{self.alpha:.4f}")
                
        return self.model
