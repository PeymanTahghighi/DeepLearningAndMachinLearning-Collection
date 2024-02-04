"""
    Re-Implementation of Visual Transformers (ViT) based on:
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). 
    An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

    Has similarity with: https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/vision_transformer

"""

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import argparse
import os
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch

def load_dataset(args):
    """ Load dataset from imagenette2, it only contains 10 classes from original ImageNet
    
    """
    aug = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize((args.resize, args.resize))
        ]
    )
    dataset_train = ImageFolder(args.train_root, transform=aug);
    dataset_test = ImageFolder(args.test_root);
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True);
    test_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False);
    return train_loader, test_loader;

class PatchEmbed(nn.Module):
    """Split image into patches and then embed them.

    Parameters
    ----------
    img_size : int
        Size of the image (it is a square).

    patch_size : int
        Size of the patch (it is a square).

    in_chans : int
        Number of input channels.

    embed_dim : int
        The emmbedding dimension.

    Attributes
    ----------
    n_patches : int
        Number of patches inside of our image.

    proj : nn.Conv2d
        Convolutional layer that does both the splitting into patches
        and their embedding.
    """
    def __init__(self, 
                 img_size, 
                 patch_size, 
                 in_chan, 
                 embed_dim) -> None:
        super().__init__();
        self.proj = nn.Conv2d(in_chan, embed_dim, patch_size, patch_size);
        self.img_size = img_size;
        self.patch_size = patch_size;
        self.n_patches = (img_size // patch_size)**2;
        self.embed_dim = embed_dim;

    def forward(self, x):
        B,C,H,W = x.shape;
        proj = self.proj(x);
        proj = proj.flatten(2);
        proj = proj.transpose(1, 2);
        return proj;

class Attention(nn.Module):
    """Attention mechanism.

    Parameters
    ----------
    dim : int
        The input and out dimension of per token features.

    n_heads : int
        Number of attention heads.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    attn_p : float
        Dropout probability applied to the query, key and value tensors.

    proj_p : float
        Dropout probability applied to the output tensor.


    Attributes
    ----------
    scale : float
        Normalizing consant for the dot product.

    qkv : nn.Linear
        Linear projection for the query, key and value.

    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention
        heads and maps it into a new space.

    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """
    def __init__(self, 
                 dim, 
                 n_heads = 12, 
                 qkv_bias = True, 
                 attn_p = 0., 
                 proj_p = 0.) -> None:
        super().__init__();
        self.dim = dim;
        self.n_heads = n_heads;
        self.qkv_bias = qkv_bias;
        self.attn_p = attn_p;
        self.proj_p = proj_p;
        self.head_dim = dim // n_heads;
        self.scale = self.head_dim ** (-0.5)

        self.qkv = nn.Linear(dim, dim*3, bias = qkv_bias);
        self.attn_drop = nn.Dropout(attn_p);
        self.proj_p_drop = nn.Dropout(proj_p);
        self.proj = nn.Linear(dim, dim);

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape;
        if dim!=self.dim:
            raise ValueError;

        qkv = self.qkv(x);
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim);
        qkv = qkv.permute(2, 0, 3, 1, 4);

        q,k,v = qkv[0], qkv[1], qkv[2];

        attn = q@k.transpose(-2, -1) * self.scale;
        attn = torch.softmax(attn, dim = -1);
        attn = self.attn_drop(attn);
        weighted_avg = attn @ v;
        weighted_avg = weighted_avg.transpose(1, 2);
        weighted_avg = weighted_avg.flatten(2);

        x = self.proj(weighted_avg);
        x = self.proj_p_drop(x);

        return x;

class MLP(nn.Module):
    """Multilayer perceptron.

    Parameters
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of output features. usually in_features*4

    p : float
        Dropout probability.

    Attributes
    ----------
    fc : nn.Linear
        The First linear layer.

    act : nn.GELU
        GELU activation function.

    fc2 : nn.Linear
        The second linear layer.

    drop : nn.Dropout
        Dropout layer.
    """
    def __init__(self, input_features, hidden_features, p=0.) -> None:
        super().__init__();
        self.fc1 = nn.Linear(input_features, hidden_features);
        self.act1 = nn.GELU();
        self.fc2 = nn.Linear(hidden_features, input_features);
        self.drop = nn.Dropout(p);


    def forward(self, x):
        x = self.fc1(x);
        x = self.act1(x);
        x = self.drop(x);
        x = self.fc2(x);
        x = self.drop(x);
        return x;

class Block(nn.Module):
    """Transformer block.

    Parameters
    ----------
    dim : int
        Embeddinig dimension.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module with respect
        to `dim`.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probability.

    Attributes
    ----------
    norm1, norm2 : LayerNorm
        Layer normalization.

    attn : Attention
        Attention module.

    mlp : MLP
        MLP module.
    """
    def __init__(self, dim, n_heads = 12, mlp_ratio = 4, qkv_bias = True, p = 0., atten_p = 0.) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps = 1e-6);
        self.attn = Attention(
            dim,
            n_heads,
            qkv_bias,
            attn_p=atten_p,
            proj_p=p
        )

        self.norm2 = nn.LayerNorm(dim, eps = 1e-6);
        self.mlp = MLP(
            dim,
            dim * mlp_ratio,
             p = p
        )

    def forward(self, x):
        out = x + self.attn(self.norm1(x));
        out = out + self.mlp(self.norm2(out));
        return out;



class ViT(nn.Module):
    """Simplified implementation of the Vision transformer.

    Parameters
    ----------
    img_size : int
        Both height and the width of the image (it is a square).

    patch_size : int
        Both height and the width of the patch (it is a square).

    in_chans : int
        Number of input channels.

    n_classes : int
        Number of classes.

    embed_dim : int
        Dimensionality of the token/patch embeddings.

    depth : int
        Number of blocks.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension of the `MLP` module.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probability.

    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of `PatchEmbed` layer.

    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has `embed_dim` elements.

    pos_emb : nn.Parameter
        Positional embedding of the cls token + all the patches.
        It has `(n_patches + 1) * embed_dim` elements.

    pos_drop : nn.Dropout
        Dropout layer.

    blocks : nn.ModuleList
        List of `Block` modules.

    norm : nn.LayerNorm
        Layer normalization.
    """
    def __init__(self, args) -> None:
        super().__init__();
        self.patch_embed = PatchEmbed(
            args.img_size,
            args.patch_size,
            args.in_channel,
            args.embed_dim
        )

        self.cls_token = nn.Parameter(torch.zeros(1,1, args.embed_dim));
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+ self.patch_embed.n_patches, args.embed_dim))
        self.pos_drop = nn.Dropout(args.p)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim = args.embed_dim,
                    n_heads = args.n_heads,
                    mlp_ratio= args.mlp_ratio,
                    qkv_bias= args.qkv_bias,
                    p = args.p,
                    atten_p=args.atten_p
                )
                for _ in range(args.n_layers)
            ]
        )

        self.norm = nn.LayerNorm(args.embed_dim, eps=1e-6);
        self.head = nn.Linear(args.embed_dim, args.n_classes);
    

    def forward(self, x):
        tokens = self.patch_embed(x);
        cls_token = self.cls_token.expand(x.shape[0], -1, -1);
        tokens = torch.cat([cls_token, tokens], dim = 1);
        tokens = tokens + self.pos_embed;
        tokens = self.pos_drop(tokens);
        for b in self.blocks:
            tokens = b(tokens);
        self.norm(tokens);
        cls_token_final = tokens[:, 0];
        out = self.head(cls_token_final);
        return out;

def train(args, model, train_loader, test_loader, optimizer):
    for epoch in range(args.epochs):
        train_step(args, epoch, model, train_loader, optimizer);
        valid_step(args, epoch, model, test_loader);
        

def train_step(args, epoch, model, loader, optimizer):
    model.train();
    print(('\n' + '%10s'*3) %('Epoch', 'Loss', 'Acc'));
    pbar = tqdm(enumerate(loader), total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    
    epoch_loss = [];
    epoch_acc = [];
    for batch_idx, (img, lbl) in pbar:
        img, lbl = img.to(args.device), lbl.to(args.device);
        out = model(img);
        loss = F.cross_entropy(out, lbl);
        out = F.softmax(out, dim = 1);
        out = torch.argmax(out, dim = 1);
        cr = torch.zeros(args.batch_size);
        cr = torch.where(out == lbl, 1, 0);
        acc = torch.sum(cr) / args.batch_size;
        epoch_acc.append(acc.item());
        epoch_loss.append(loss.item());
        pbar.set_description(('%10s' + '%10.4g'*2) %(epoch, np.mean(epoch_loss), np.mean(epoch_acc)));
        loss.backward();
        optimizer.step();
        model.zero_grad(set_to_none = True);

def valid_step(args, epoch, model, loader):
    model.eval();
    with torch.no_grad():
        print(('\n' + '%10s'*3) %('Epoch', 'Loss', 'Acc'));
        pbar = tqdm(enumerate(loader), total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        epoch_loss = [];
        epoch_acc = [];
        for batch_idx, (img, lbl) in pbar:
            img, lbl = img.to(args.device), lbl.to(args.device);
            out = model(img);
            loss = F.cross_entropy(out, lbl);
            out = F.softmax(out, dim = 1);
            out = torch.argmax(out, dim = 1);
            cr = torch.zeros(args.batch_size);
            cr = torch.where(out == lbl, 1, 0);
            acc = torch.sum(cr) / args.batch_size;
            epoch_acc.append(acc.item());
            epoch_loss.append(loss.item());
            pbar.set_description(('%10s' + '%10.4g'*2) %(epoch, np.mean(epoch_loss), np.mean(epoch_acc)));
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'ViT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-train-root', default = os.path.join('../datasets', 'imagenette2', 'imagenette2', 'train'), help='Root training images');
    parser.add_argument('-test-root', default = os.path.join('../datasets', 'imagenette2', 'imagenette2', 'val'), help='Root training images');
    parser.add_argument('-batch-size', default=256);
    parser.add_argument('-num-workers', default=0, help='number of workers for data loader');
    parser.add_argument('-epochs', default=10, help='number training epochs');
    parser.add_argument('-resize', default=224)
    parser.add_argument('-lr', default=3e-4)
    parser.add_argument('-device', default='cuda')
    parser.add_argument('-img_size', default=384)
    parser.add_argument('-patch-size', default=16)
    parser.add_argument('-in_channel', default=3)
    parser.add_argument('-embed-dim', default=768)
    parser.add_argument('-n_heads', default=8)
    parser.add_argument('-mlp_ratio', default=4)
    parser.add_argument('-qkv_bias', default=True)
    parser.add_argument('-p', default=False)
    parser.add_argument('-atten-p', default=False)
    parser.add_argument('-n_layers', default=12)
    parser.add_argument('-n_classes', default=1000)
    args = parser.parse_args();


    train_loader, test_loader = load_dataset(args);
    model = ViT(args).to(args.device);

    opimizer = optim.AdamW(model.parameters(), lr = args.lr);
    train(args, model, train_loader, test_loader, opimizer);