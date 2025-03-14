from torch.nn import Module


class MagicMotion(Module):
    def __init__(self, transformer, controlnet):
        super().__init__()
        self.transformer = transformer
        self.controlnet = controlnet
