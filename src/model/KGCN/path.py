import os

class Path:
    def __init__(self, dataset):
        self.data = f'../../data/{dataset}/'
        self.misc = f'../model/KGCN/misc/{dataset}/'
        self.emb = f'../model/KGCN/misc/{dataset}/emb/'
        self.output = f'../model/KGCN/output/{dataset}/'
        
        self.check_dir(f'../model/KGCN/output')
        self.check_dir(f'../model/KGCN/misc')
        self.check_dir(self.data)
        self.check_dir(self.misc)
        self.check_dir(self.emb)
        self.check_dir(self.output)

    def check_dir(self, p):
        if not os.path.isdir(p):
            os.mkdir(p)

class Path_tag:
    def __init__(self, args):
        self.data = f'../../data/{args.dataset}/'
        self.misc = f'../../misc/{args.dataset}/'
        self.emb = f'../../misc/{args.dataset}/emb/'
        self.output = f'../../output/KGCN/{args.dataset}/{args.exp_tag}/'

        self.check_dir(f'../../output/KGCN/')
        self.check_dir(f'../../output/KGCN/{args.dataset}/')
        self.check_dir(self.data)
        self.check_dir(self.misc)
        self.check_dir(self.emb)
        self.check_dir(self.output)

    def check_dir(self, p):
        if not os.path.isdir(p):
            os.mkdir(p)
