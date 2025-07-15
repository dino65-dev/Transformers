import torch
def rotate_half(x):
        # it's a 90° rotation , if you think x as a complex number input then , x-->  a+bi then after 90° rotation it will be -b+ai
        x1, x2 = x[...,:x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

@torch.jit.script
def apply_rotary_pos_emb(x,cos,sin):
        #it applies Euler formula : e^(iθ) = cos(θ) + i·sin(θ) that causes (q * cos) + (rotate_half(q) * sin) is implementing: q·cos(θ) + i·q·sin(θ)
        # rotate_half is for to make the q&k (iota)imaginary part
        return (x * cos) + (rotate_half(x) * sin)
