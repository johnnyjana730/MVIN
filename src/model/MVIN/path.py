import os

class Path:
    def __init__(self, args):
        self.data = f'../../data/{args.dataset}/'
        self.misc = f'../model/MVIN/misc/{args.dataset}/'
        self.emb = f'../model/MVIN/misc/{args.dataset}/emb/'
        self.case_st = f'../../case_st/{args.dataset}/'
        self.output = f'../model/MVIN/output/{args.dataset}/MVIN_p0{args.p_hop}_h_{args.h_hop}/no_sw'

        if args.abla_exp == True:
            self.output = f'../model/MVIN/output/{args.dataset}/abla_exp/'
        
        if args.top_k == True:
            self.output = f"{self.output}/topk/"

        self.check_dir(f'../model/MVIN/output/')
        self.check_dir(f'../model/MVIN/output/{args.dataset}/')
        self.check_dir(f'../model/MVIN/output/{args.dataset}/MVIN_p0{args.p_hop}_h_{args.h_hop}/')
        self.check_dir(f'../model/MVIN/output/{args.dataset}/MVIN_p0{args.p_hop}_h_{args.h_hop}/no_sw/')

        self.check_dir(self.data)
        self.check_dir(f'../model/MVIN/misc/')
        self.check_dir(self.misc)
        self.check_dir(self.emb)
        self.check_dir(f'../../case_st/')
        self.check_dir(self.case_st)
        self.check_dir(self.output)

    def check_dir(self, p):
        if not os.path.isdir(p):
            os.mkdir(p)

class Path_SW:
    def __init__(self, args):
        self.data = f'../../data/{args.dataset}/'
        self.misc = f'../model/MVIN/misc/{args.dataset}/'
        self.emb = f'../model/MVIN/misc/{args.dataset}/emb/'
        self.case_st = f'../../case_st/{args.dataset}/'
        self.output = f'../model/MVIN/output/{args.dataset}/MVIN_p0{args.p_hop}_h_{args.h_hop}/sw/'

        if args.abla_exp == True:
            self.output = f'../model/MVIN/output/{args.dataset}/abla_exp/'
        
        if args.top_k == True:
            self.output = f"{self.output}/topk/"

        self.check_dir(f'../model/MVIN/output/')
        self.check_dir(f'../model/MVIN/output/{args.dataset}/')
        self.check_dir(f'../model/MVIN/output/{args.dataset}/MVIN_p0{args.p_hop}_h_{args.h_hop}/')
        self.check_dir(f'../model/MVIN/output/{args.dataset}/MVIN_p0{args.p_hop}_h_{args.h_hop}/sw/')

        self.check_dir(self.data)
        self.check_dir(f'../model/MVIN/misc/')
        self.check_dir(self.misc)
        self.check_dir(self.emb)
        self.check_dir(f'../../case_st/')
        self.check_dir(self.case_st)
        self.check_dir(self.output)

    def check_dir(self, p):
        if not os.path.isdir(p):
            os.mkdir(p)
