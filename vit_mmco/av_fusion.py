import torch
import torch.nn as nn
class AV_Fuison(torch.nn.Module):
    def __init__(self,**args):
        super().__init__()
        embed_dim = 1024
        self.U_v = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
        )

        self.U_a = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
        )
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
                                       nn.Dropout(0.5),
                                       nn.Sigmoid())
    def forward(self,afeat,vfeat):
        video=torch.randn(10,320,1024)
        audio=torch.randn(10,320,1024)
        for i in range(afeat.shape[0]):
            video_residual = vfeat[i]
            v = self.U_v(vfeat[i])
            audio_residual = afeat[i]
            a = self.U_a(afeat[i])
            merged = torch.mul(v + a, 0.5) 

            a_trans = audio_residual
            v_trans = video_residual

            video[i] = nn.Tanh()(a_trans + merged)
            audio[i] = nn.Tanh()(v_trans + merged)
        a_vfeat = torch.mul(video + audio, 0.5)
        a_vfeat = a_vfeat.transpose(-1,-2)
        a_v_atn = self.attention(a_vfeat)
        return a_v_atn,a_vfeat

afeat = torch.randn(10,320,1024)
vfeat = torch.randn(10,320,1024)
av_fusion = AV_Fuison()
a_v_atn,a_vfeat = av_fusion(afeat,vfeat)
print(a_v_atn.shape)