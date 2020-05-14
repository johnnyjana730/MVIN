import numpy as np

def parameter_env(args):
    if args.ablation == 'all':
        args.SW = 1
        args.User_orient = 1
        args.User_orient_rela = 1
        args.User_orient_kg_eh = 1
        args.PS_O_ft = 1
        args.wide_deep = 1
        args.PS_only = 0
        args.HO_only = 0
    elif args.ablation == 'no_sw':
        args.SW = 0
        args.User_orient = 1
        args.User_orient_rela = 1
        args.User_orient_kg_eh = 1
        args.PS_O_ft = 1
        args.wide_deep = 1
        args.PS_only = 0
        args.HO_only = 0
    elif args.ablation == 'no_kg_eh_uo':
        args.SW = 1
        args.User_orient = 1
        args.User_orient_rela = 1
        args.User_orient_kg_eh = 0
        args.PS_O_ft = 1
        args.wide_deep = 1
        args.PS_only = 0
        args.HO_only = 0
    elif args.ablation == 'no_kg_eh_uo_sw':
        args.SW = 1
        args.User_orient = 1
        args.User_orient_rela = 1
        args.User_orient_kg_eh = 0
        args.PS_O_ft = 1
        args.wide_deep = 1
        args.PS_only = 0
        args.HO_only = 0
    elif args.ablation == 'no_uo_and_no_kg_eh_uo':
        args.SW = 1
        args.User_orient = 0
        args.User_orient_rela = 1
        args.User_orient_kg_eh = 0
        args.PS_O_ft = 1
        args.wide_deep = 1
        args.PS_only = 0
        args.HO_only = 0
    elif args.ablation == 'no_uo_and_no_kg_eh_uo_sw':
        args.SW = 1
        args.User_orient = 0
        args.User_orient_rela = 1
        args.User_orient_kg_eh = 0
        args.PS_O_ft = 1
        args.wide_deep = 1
        args.PS_only = 0
        args.HO_only = 0
    elif args.ablation == 'no_uor_and_no_kg_eh_uo':
        args.SW = 1
        args.User_orient = 1
        args.User_orient_rela = 0
        args.User_orient_kg_eh = 0
        args.PS_O_ft = 1
        args.wide_deep = 1
        args.PS_only = 0
        args.HO_only = 0
    elif args.ablation == 'no_uor_and_no_kg_eh_uo_sw':
        args.SW = 1
        args.User_orient = 1
        args.User_orient_rela = 0
        args.User_orient_kg_eh = 0
        args.PS_O_ft = 1
        args.wide_deep = 1
        args.PS_only = 0
        args.HO_only = 0
    elif args.ablation == 'no_uo':
        args.SW = 1
        args.User_orient = 0
        args.User_orient_rela = 1
        args.User_orient_kg_eh = 1
        args.PS_O_ft = 1
        args.wide_deep = 1
        args.PS_only = 0
        args.HO_only = 0
    elif args.ablation == 'no_uor':
        args.SW = 1
        args.User_orient = 1
        args.User_orient_rela = 0
        args.User_orient_kg_eh = 1
        args.PS_O_ft = 1
        args.wide_deep = 1
        args.PS_only = 0
        args.HO_only = 0
    elif args.ablation == 'no_wd':
        args.SW = 1
        args.User_orient = 1
        args.User_orient_rela = 1
        args.User_orient_kg_eh = 1
        args.PS_O_ft = 1
        args.wide_deep = 0
        args.PS_only = 0
        args.HO_only = 0
    elif args.ablation == 'no_ps_o_ft':
        args.SW = 1
        args.User_orient = 1
        args.User_orient_rela = 1
        args.User_orient_kg_eh = 1
        args.PS_O_ft = 0
        args.wide_deep = 1
        args.PS_only = 0
        args.HO_only = 0
    elif args.ablation == 'ps_only':
        args.SW = 1
        args.User_orient = 1
        args.User_orient_rela = 1
        args.User_orient_kg_eh = 1
        args.PS_O_ft = 1
        args.wide_deep = 1
        args.PS_only = 1
        args.HO_only = 0
    elif args.ablation == 'ho_only':
        args.SW = 1
        args.User_orient = 1
        args.User_orient_rela = 1
        args.User_orient_kg_eh = 0
        args.PS_O_ft = 1
        args.wide_deep = 1
        args.PS_only = 0
        args.HO_only = 1
    elif args.ablation == 'ho_only_uo_kg_eh':
        args.SW = 1
        args.User_orient = 1
        args.User_orient_rela = 1
        args.User_orient_kg_eh = 1
        args.PS_O_ft = 1
        args.wide_deep = 1
        args.PS_only = 0
        args.HO_only = 1
    elif args.ablation == 'no_wd_ho_only':
        args.SW = 1
        args.User_orient = 1
        args.User_orient_rela = 1
        args.User_orient_kg_eh = 0
        args.PS_O_ft = 1
        args.wide_deep = 0
        args.PS_only = 0
        args.HO_only = 1
    elif args.ablation == 'no_uo_ho_only':
        args.SW = 1
        args.User_orient = 0
        args.User_orient_rela = 1
        args.User_orient_kg_eh = 0
        args.PS_O_ft = 1
        args.wide_deep = 1
        args.PS_only = 0
        args.HO_only = 1
    elif args.ablation == 'no_uor_ho_only':
        args.SW = 1
        args.User_orient = 1
        args.User_orient_rela = 0
        args.User_orient_kg_eh = 0
        args.PS_O_ft = 1
        args.wide_deep = 1
        args.PS_only = 0
        args.HO_only = 1
        
    args.abla_exp = (args.abla_exp == 1)
    args.SW = (args.SW == 1)
    args.User_orient = (args.User_orient == 1)
    args.User_orient_rela = (args.User_orient_rela == 1)
    args.User_orient_kg_eh = (args.User_orient_kg_eh == 1)
    args.PS_O_ft = (args.PS_O_ft == 1)
    args.wide_deep = (args.wide_deep == 1)
    args.PS_only = (args.PS_only == 1)
    args.HO_only = (args.HO_only == 1)

    print('args.ablation = ', args.ablation, 'args.SW = ', args.SW, 'args.User_orient = ', args.User_orient,  'args.User_orient_rela = ', \
           args.User_orient_rela, 'args.User_orient_kg_eh = ', args.User_orient_kg_eh ,  'args.PS_O_ft = ', args.PS_O_ft,\
          'args.wide_deep = ', args.wide_deep, 'args.PS_only = ', args.PS_only, 'args.HO_only = ', args.HO_only)
    return args