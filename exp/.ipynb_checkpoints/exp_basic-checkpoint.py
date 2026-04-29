import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, MICN, Crossformer, FiLM, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, \
    TimeXer, TimeXerWGA, TimeXerWAuto, TimeXerWProb, TimeXerWGeom, TimeXerWTOST, \
    iTransformer, iTransformerWGA, iTransformerWAuto, iTransformerWProb, iTransformerWGeom, iTransformerWTOST,\
    SimpleTM, SimpleTMWGA, SimpleTMWAuto, SimpleTMWProb, SimpleTMWGeom, SimpleTMWTOST, \
    TimeXerWGAwoPsi, TimeXerWGAwoAlpha, TimeXerWGAwoSparse, TimeXerWGAwMLP, TimeXerWGAwSA, TimeXerWGAwoOrth, \
    PatchTST, PatchTSTWGA, PatchTSTWAuto, PatchTSTWProb, PatchTSTWGeom, PatchTSTWTOST, \
    CARD, CARDWGA, CARDWAuto, CARDWProb, CARDWGeom, CARDWTOST, \
    FEDformer, FEDformerWGA, FEDformerWAuto, FEDformerWProb, FEDformerWGeom, FEDformerWTOST, \
    PAttn, PAttnWGA, PAttnWAuto, PAttnWProb, PAttnWGeom, PAttnWTOST, \
    MultiPatchFormer, MultiPatchFormerWGA, MultiPatchFormerWAuto, MultiPatchFormerWProb, MultiPatchFormerWGeom, MultiPatchFormerWTOST



class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'PAttn': PAttn,
            'PAttnWGA': PAttnWGA,
            'PAttnWAuto': PAttnWAuto,
            'PAttnWProb': PAttnWProb,
            'PAttnWGeom': PAttnWGeom,
            'PAttnWTOST': PAttnWTOST,
            'MultiPatchFormer': MultiPatchFormer,
            'MultiPatchFormerWGA': MultiPatchFormerWGA,
            'MultiPatchFormerWAuto': MultiPatchFormerWAuto,
            'MultiPatchFormerWProb': MultiPatchFormerWProb,
            'MultiPatchFormerWGeom': MultiPatchFormerWGeom,
            'MultiPatchFormerWTOST': MultiPatchFormerWTOST,
            'CARD': CARD,
            'CARDWGA': CARDWGA,
            'CARDWAuto': CARDWAuto,
            'CARDWProb': CARDWProb,
            'CARDWGeom': CARDWGeom,
            'CARDWTOST': CARDWTOST,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'FEDformerWGA': FEDformerWGA,
            'FEDformerWAuto': FEDformerWAuto,
            'FEDformerWProb': FEDformerWProb,
            'FEDformerWGeom': FEDformerWGeom,
            'FEDformerWTOST': FEDformerWTOST,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'PatchTSTWGA': PatchTSTWGA,
            'PatchTSTWAuto': PatchTSTWAuto,
            'PatchTSTWProb': PatchTSTWProb,
            'PatchTSTWGeom': PatchTSTWGeom,
            'PatchTSTWTOST': PatchTSTWTOST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'iTransformerWGA': iTransformerWGA,
            'iTransformerWAuto': iTransformerWAuto,
            'iTransformerWProb': iTransformerWProb,
            'iTransformerWGeom': iTransformerWGeom,
            'iTransformerWTOST': iTransformerWTOST,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'TimeXer': TimeXer,
            'TimeXerWGA': TimeXerWGA,
            'TimeXerWGAwoPsi': TimeXerWGAwoPsi,
            'TimeXerWGAwoAlpha': TimeXerWGAwoAlpha,
            'TimeXerWGAwoSparse': TimeXerWGAwoSparse,
            'TimeXerWGAwMLP': TimeXerWGAwMLP,
            'TimeXerWGAwSA': TimeXerWGAwSA,
            'TimeXerWGAwoOrth': TimeXerWGAwoOrth,
            'TimeXerWAuto': TimeXerWAuto,
            'TimeXerWProb': TimeXerWProb,
            'TimeXerWGeom': TimeXerWGeom,
            'TimeXerWTOST': TimeXerWTOST,
            'SimpleTM': SimpleTM,
            'SimpleTMWGA': SimpleTMWGA,
            'SimpleTMWAuto': SimpleTMWAuto,
            'SimpleTMWProb': SimpleTMWProb,
            'SimpleTMWGeom': SimpleTMWGeom,
            'SimpleTMWTOST': SimpleTMWTOST,
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
