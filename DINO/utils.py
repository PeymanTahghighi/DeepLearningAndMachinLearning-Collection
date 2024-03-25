from typing import Any
import torch
import torch.nn as nn
import torch.functional as F
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as plt
class DataAugmetation:
    def __init__(self, 
                 global_crops_scale = (0.4, 1.0), 
                 local_crops_scale = (0.05, 0.4), 
                 n_local_crops = 8, 
                 size = 224):
        self.n_local_crops = n_local_crops;
        RandomGaussianBlur = lambda p: transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2))], 
            p = p
        )

        flip_and_jitter = transforms.Compose(
            [
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.1
                    )
                ]
            ),
            transforms.RandomGrayscale(0.2)
            ]
        )
        
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )

        self.global1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size,
                    scale = global_crops_scale,
                    interpolation=Image.BICUBIC
                ),
                flip_and_jitter,
                RandomGaussianBlur(0.1),
                normalize
            ]
        )

        self.global2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size,
                    scale = global_crops_scale,
                    interpolation=Image.BICUBIC
                ),
                flip_and_jitter,
                RandomGaussianBlur(0.1),
                transforms.RandomSolarize(170, 0.2),
                normalize
            ]
        )

        self.local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size,
                    scale = local_crops_scale,
                    interpolation=Image.BICUBIC
                ),
                flip_and_jitter,
                RandomGaussianBlur(0.5),
                normalize
            ]
        )

    def __call__(self, img):
        all_crops = [];
        all_crops.append(self.global1(img));
        all_crops.append(self.global2(img));
        all_crops.extend([self.local(img) for _ in range(self.n_local_crops)])
        return all_crops;

class Head(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 hidden_dim = 2048, 
                 bottleneck_dim = 256, 
                 n_layers = 3, 
                 norm_last_layer = False):
        super().__init__();
        if n_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim);
        else:
            layers = [nn.Linear(in_dim, hidden_dim)];
            layers.append(nn.GELU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim));
                layers.append(nn.GELU());

            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers);
        self.apply(self._init_weights);
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias = False)
        )

        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False;

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std  = 0.02);
            if m.bias is not None:
                nn.init.constant_(m.bias, 0);

    def forward(self, x):
        x = self.mlp(x);
        x = nn.functional.normalize(x, dim = -1, p = 2)
        x = self.last_layer(x);
        return x;

class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, new_head) -> None:
        super().__init__()
        backbone.head = nn.Identity()
        self.backbone = backbone;
        self.head = new_head;

    def forward(self, x):
        n_crops = len(x);
        concatenated = torch.cat(x, dim = 0);
        cls_embedding = self.backbone(concatenated);
        logits = self.head(cls_embedding);
        chunks = logits.chunk(n_crops);
        return chunks;


class Loss(nn.Module):
    def __init__(self, 
                    out_dim, 
                    teacher_temp = 0.04, 
                    student_temp = 0.1,
                    center_momentum = 0.9):
        super().__init__()
        self.student_temp = student_temp;
        self.teacher_temp = teacher_temp;
        self.center_momentum = center_momentum;
        self.register_buffer("center", torch.zeros(1, out_dim));

    def forward(self, student_output, teacher_output):
        student_temp = [s / self.student_temp for s in student_output];
        teacher_temp = [(t-self.center) / self.teacher_temp for t in teacher_output];
        student_sm = [torch.log_softmax(s, dim=-1) for s in student_temp];
        teacher_sm = [torch.softmax(t, dim=-1).detach() for t in teacher_temp]
        total_loss = 0;
        n_loss_terms = 0;

        for t_ix,t in enumerate(teacher_sm):
            for s_ix, s in enumerate(student_sm):
                if t_ix == s_ix:
                    continue;

                loss = torch.sum(-t*s, dim= -1)
                total_loss += loss.mean();
                n_loss_terms += 1;

        total_loss /= n_loss_terms;

        self.update_center(teacher_output);
        return total_loss;

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.cat(teacher_output).mean(dim = 0, keepdim=True);
        self.center = self.center * self.center_momentum + batch_center * (1-self.center_momentum);

def clip_gradient(model, clip = 2.0):
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)