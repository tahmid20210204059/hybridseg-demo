import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, ic, oc, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(oc, eps=1e-5),
            nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)


class _RB(nn.Module):
    def __init__(self, ic, oc, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ic, oc, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(oc, eps=1e-5), nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False),
            nn.BatchNorm2d(oc, eps=1e-5))
        self.down = nn.Sequential(
            nn.Conv2d(ic, oc, 1, stride=stride, bias=False),
            nn.BatchNorm2d(oc, eps=1e-5)) if (stride != 1 or ic != oc) else None
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.body(x) + (self.down(x) if self.down else x))


class ResNet34Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, eps=1e-5), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.e1 = self._make(64,  64,  3, 1)
        self.e2 = self._make(64,  128, 4, 2)
        self.e3 = self._make(128, 256, 6, 2)
        self.e4 = self._make(256, 512, 3, 2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def _make(self, ic, oc, n, stride):
        return nn.Sequential(_RB(ic, oc, stride), *[_RB(oc, oc) for _ in range(1, n)])
    def forward(self, x):
        s = self.stem(x)
        e1 = self.e1(s); e2 = self.e2(e1)
        e3 = self.e3(e2); e4 = self.e4(e3)
        return e1, e2, e3, e4


class S6Block(nn.Module):
    def __init__(self, d_model, d_state=8):
        super().__init__()
        di = d_model
        self.di, self.ds = di, d_state
        self.in_proj  = nn.Linear(d_model, di * 2, bias=False)
        self.conv1d   = nn.Conv1d(di, di, 3, padding=2, groups=di, bias=True)
        self.x_proj   = nn.Linear(di, di + d_state * 2, bias=False)
        self.dt_proj  = nn.Linear(di, di)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(di, 1)
        self.A_log  = nn.Parameter(torch.log(A))
        self.D      = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)
    def _scan(self, u, dt, A, B, C):
        u, dt, B, C = u.float(), dt.float(), B.float(), C.float()
        Bs, L, d = u.shape
        h = torch.zeros(Bs, d, self.ds, device=u.device)
        ys = []
        for t in range(L):
            dt_t = torch.sigmoid(dt[:, t]) * 0.5
            dA_t = torch.exp(-torch.abs(A[None]) * dt_t.unsqueeze(-1))
            h    = dA_t * h + dt_t.unsqueeze(-1) * B[:, t, None, :] * u[:, t, :, None]
            h    = h.clamp(-100.0, 100.0)
            ys.append((h * C[:, t, None, :]).sum(-1))
        return torch.stack(ys, 1)
    def forward(self, x):
        orig = x.dtype; x = x.float(); res = x
        xi, z = self.in_proj(x).chunk(2, -1)
        xi = self.conv1d(xi.transpose(1,2))[..., :xi.shape[1]].transpose(1,2)
        xi = F.silu(xi)
        proj = self.x_proj(xi)
        dt   = self.dt_proj(proj[..., :self.di])
        Bs   = proj[..., self.di:self.di + self.ds]
        Cs   = proj[..., self.di + self.ds:]
        y    = self._scan(xi, dt, self.A_log.float(), Bs, Cs)
        y    = torch.nan_to_num(y, nan=0.0, posinf=100., neginf=-100.)
        y    = (y + xi * self.D[None, None]) * F.silu(z)
        return self.norm(self.out_proj(y) + res).to(orig)


class VMambaSSMBridge(nn.Module):
    def __init__(self, ch, d_state=8):
        super().__init__()
        self.s6_rr = S6Block(ch, d_state)
        self.s6_rl = S6Block(ch, d_state)
        self.s6_cr = S6Block(ch, d_state)
        self.s6_cl = S6Block(ch, d_state)
        self.fuse  = nn.Sequential(nn.Conv2d(ch, ch, 1, bias=False), nn.BatchNorm2d(ch, eps=1e-5))
        self.gate  = nn.Parameter(torch.tensor(0.1))
    def _row(self, x, blk, rev=False):
        B, C, H, W = x.shape
        s = x.permute(0,2,3,1).reshape(B*H, W, C)
        if rev: s = s.flip(1)
        o = blk(s)
        if rev: o = o.flip(1)
        return o.reshape(B,H,W,C).permute(0,3,1,2)
    def _col(self, x, blk, rev=False):
        B, C, H, W = x.shape
        s = x.permute(0,3,2,1).reshape(B*W, H, C)
        if rev: s = s.flip(1)
        o = blk(s)
        if rev: o = o.flip(1)
        return o.reshape(B,W,H,C).permute(0,3,2,1)
    def forward(self, x):
        y = (self._row(x, self.s6_rr) + self._row(x, self.s6_rl, True) +
             self._col(x, self.s6_cr) + self._col(x, self.s6_cl, True))
        y = torch.nan_to_num(y, nan=0.0, posinf=100., neginf=-100.)
        return x + self.gate.clamp(0, 1) * self.fuse(y)


def _proj(ic, oc):
    return nn.Sequential(nn.Conv2d(ic, oc, 1, bias=False), nn.BatchNorm2d(oc, eps=1e-5), nn.ReLU(inplace=True))
def _dec(ic, oc):
    return nn.Sequential(ConvBNReLU(ic, oc), ConvBNReLU(oc, oc))


class UNet3PlusDecoder(nn.Module):
    def __init__(self, enc=(64,128,256,256), k=96):
        super().__init__()
        t = k * 4
        self.d4 = nn.ModuleDict({'e1':_proj(enc[0],k),'e2':_proj(enc[1],k),'e3':_proj(enc[2],k),'e4':_proj(enc[3],k),'dec':_dec(t,256)})
        self.d3 = nn.ModuleDict({'e1':_proj(enc[0],k),'e2':_proj(enc[1],k),'e3':_proj(enc[2],k),'d4':_proj(256,k),'dec':_dec(t,128)})
        self.d2 = nn.ModuleDict({'e1':_proj(enc[0],k),'e2':_proj(enc[1],k),'d3':_proj(128,k),'d4':_proj(256,k),'dec':_dec(t,64)})
        self.d1 = nn.ModuleDict({'e1':_proj(enc[0],k),'d2':_proj(64,k),'d3':_proj(128,k),'d4':_proj(256,k),'dec':_dec(t,32)})
    def _rs(self, x, ref):
        return F.interpolate(x, ref.shape[2:], mode='bilinear', align_corners=False) if x.shape[2:] != ref.shape[2:] else x
    def forward(self, e1, e2, e3, e4):
        m=self.d4; f4=m['dec'](torch.cat([m['e1'](F.adaptive_avg_pool2d(e1,e4.shape[2:])),m['e2'](F.adaptive_avg_pool2d(e2,e4.shape[2:])),m['e3'](F.adaptive_avg_pool2d(e3,e4.shape[2:])),m['e4'](e4)],1))
        m=self.d3; f3=m['dec'](torch.cat([m['e1'](F.adaptive_avg_pool2d(e1,e3.shape[2:])),m['e2'](F.adaptive_avg_pool2d(e2,e3.shape[2:])),m['e3'](e3),m['d4'](self._rs(f4,e3))],1))
        m=self.d2; f2=m['dec'](torch.cat([m['e1'](F.adaptive_avg_pool2d(e1,e2.shape[2:])),m['e2'](e2),m['d3'](self._rs(f3,e2)),m['d4'](self._rs(f4,e2))],1))
        m=self.d1; f1=m['dec'](torch.cat([m['e1'](e1),m['d2'](self._rs(f2,e1)),m['d3'](self._rs(f3,e1)),m['d4'](self._rs(f4,e1))],1))
        return f1, f2, f3, f4


class SegHead(nn.Module):
    def __init__(self, ic, nc=1):
        super().__init__()
        self.net = nn.Sequential(ConvBNReLU(ic, ic), nn.Conv2d(ic, nc, 1))
    def forward(self, x, sz):
        return F.interpolate(self.net(x), sz, mode='bilinear', align_corners=False)


class BoundaryHead(nn.Module):
    def __init__(self, ic):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-5), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16, eps=1e-5), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1))
    def forward(self, x, sz):
        return F.interpolate(self.net(x), sz, mode='bilinear', align_corners=False)


class HybridSegModel(nn.Module):
    def __init__(self, num_classes=1, d_state=8, ssm_warmup=5, in_channels=1):
        super().__init__()
        self.ssm_warmup = ssm_warmup
        self.encoder = ResNet34Encoder(in_channels=in_channels)
        self.proj    = nn.Sequential(nn.Conv2d(512,256,1,bias=False), nn.BatchNorm2d(256,eps=1e-5), nn.ReLU(inplace=True))
        self.ssm     = VMambaSSMBridge(256, d_state)
        self.decoder = UNet3PlusDecoder(enc=(64,128,256,256), k=96)
        self.seg     = SegHead(32, num_classes)
        self.bnd     = BoundaryHead(32)
        self.aux2    = nn.Conv2d(64,  1, 1)
        self.aux3    = nn.Conv2d(128, 1, 1)
        self.aux4    = nn.Conv2d(256, 1, 1)
    def forward(self, x, epoch=999):
        H, W = x.shape[2:]
        e1, e2, e3, e4 = self.encoder(x)
        e4 = self.proj(e4)
        if epoch >= self.ssm_warmup:
            e4 = self.ssm(e4)
            e4 = torch.nan_to_num(e4, nan=0.0, posinf=100., neginf=-100.)
        f1, f2, f3, f4 = self.decoder(e1, e2, e3, e4)
        seg = self.seg(f1, (H, W))
        bnd = self.bnd(f1, (H, W))
        return seg, bnd


# ── Model configs for each dataset ──
MODEL_CONFIGS = {
    "🫁 Chest X-Ray (Montgomery)": {
        "weights": "chest_best_weights.pth",
        "in_channels": 3,
        "img_size": 256,
        "desc": "Lung segmentation from chest X-ray images"
    },
    "🔬 Polyp (Kvasir-SEG)": {
        "weights": "polyp_best_weights.pth",
        "in_channels": 1,
        "img_size": 352,
        "desc": "Polyp segmentation from colonoscopy images"
    },
    "🩻 Breast Ultrasound (BUSI)": {
        "weights": "busi_best_weights.pth",
        "in_channels": 1,
        "img_size": 384,
        "desc": "Breast lesion segmentation from ultrasound images"
    },
}
