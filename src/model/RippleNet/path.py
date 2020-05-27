import os

class Path:
    def __init__(self, dataset):
        self.data = f'../../data/{dataset}/'
        self.misc = f'../model/RippleNet/misc/{dataset}/'
        self.emb = f'../model/RippleNet/misc/{dataset}/emb/'
        self.output = f'../model/RippleNet/output/{dataset}'

        self.check_dir(f'../model/RippleNet/output')
        self.check_dir(f'../model/RippleNet/misc')
        self.check_dir(self.data)
        self.check_dir(self.misc)
        self.check_dir(self.emb)
        self.check_dir(self.output)

    def check_dir(self, p):
        if not os.path.isdir(p):
            os.mkdir(p)
