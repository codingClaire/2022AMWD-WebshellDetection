import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--with_std', type=int, default=1)
parser.add_argument('--outdir', type=str, default='./figures')
parser.add_argument('--filename', type=str, default='output')
parser.add_argument('--filetype', type=str, default='png')
args = parser.parse_args()

ksteps = [10, 50, 150, 250]
y = [78.46, 78.30, 78.46, 78.39]
y_std = [0.27, 0.32, 0.27, 0.28]

alpha=0.9
color_dict = {'red':    '#D62728',
              'green':  '#2CA02C',
              'blue':   '#1F77B4',
              'orange': '#FF7F0E',
              'brown':  '#8C564B',
              'purple': '#9467BD'}
# marker_dict = {'cora': 'o', 'citeseer': '*', 'email': 'd', 'acm': 's'}
# linestyle_dict = {'cora': 'solid', 'citeseer': 'solid', 'email': 'solid', 'acm': 'solid'}
# label_dict = {'drop_prt':'Drop rate', 'temperature':'Temperature', 'ori_w': r'$\gamma$', 'aug_w':r'$\delta$', 
#               'cor_w':r'$\eta$', 'edge_w1':r'$\alpha$', 'edge_w2':r'$\beta$', 'temperature2':'Temperature'}  
linewidth = 8
markersize = 18
fontsize = 72
ticks_fontsize = fontsize -9
legend_fontsize = fontsize -9
plt.figure(figsize=(21,15), dpi=80)
std_alpha = 0.1

plt.plot(range(len(ksteps)), y, color=color_dict['green'], marker='o', 
            markersize=markersize, linestyle='solid', alpha=alpha, label='acc', linewidth=linewidth)
if args.with_std:
    r1 = list(map(lambda x: x[0]-x[1], zip(y, y_std)))#上方差
    r2 = list(map(lambda x: x[0]+x[1], zip(y, y_std)))#下方差
    plt.fill_between(range(len(ksteps)), r1, r2, color=color_dict['green'], alpha=std_alpha)
plt.xlabel('ksteps', fontsize=fontsize, labelpad=-2)
plt.ylabel('Accuracy', fontsize=fontsize)
plt.xticks(range(len(ksteps)), ksteps, fontsize=ticks_fontsize, rotation=20)
plt.yticks(fontsize=ticks_fontsize)
plt.tight_layout()
import os
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
plt.savefig(f'{args.outdir}/{args.filename}.{args.filetype}')