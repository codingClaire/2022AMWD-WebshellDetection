import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--with_std', type=int, default=1)
parser.add_argument('--datafile', type=str, required=True, help='rslt data file')
parser.add_argument('--readtype', type=str, required=True, help='gat, gcn')
parser.add_argument('--outdir', type=str, default='./figures')
parser.add_argument('--filename', type=str, default='', help='set the name of the output figure, default for similar to input filename')
parser.add_argument('--filetype', type=str, default='png')
args = parser.parse_args()

# read data to plot
y_avg = {}
y_std = {}
configs = []
metrics = []
if args.readtype == 'gat':
    with open(f"rslt_data/{args.datafile}", "r") as fin:
        line_cnt = 0
        for line in fin.readlines():
            items = line.strip('\n').split('Avg.')
            # config = (items[0].split('_')[1]+'+'+items[0].split('_')[2]).strip(',')
            config = items[0].strip(' ').strip(',')
            configs.append(config)
            for i in range(1, len(items)):
                metric_rslt = items[i].split(':')
                metric = metric_rslt[0]
                if line_cnt == 0:
                    metrics.append(metric)
                if metric not in y_avg.keys():
                    y_avg[metric] = []
                    y_std[metric] = []
                avg_std = metric_rslt[1].split('+-')
                y_avg[metric].append(float(avg_std[0])/100)
                y_std[metric].append(float(avg_std[1][:4])/100)
            line_cnt += 1
elif args.readtype == 'gcn':
    with open(f"rslt_data/{args.datafile}", "r") as fin:
        variant_cnt = 0
        for line in fin.readlines():
            line = line.strip('\n')
            if line[0] == '=':
                line = line.strip('=')
                jk_idx = line.index('JK:')
                pooling_idx = line.index('Pooling:')
                jk = line[jk_idx+3:pooling_idx].strip(' ')
                pooling = line[pooling_idx+len('Pooling:'):].strip(' ')
                config = 'JK:'+jk+'+Pooling:'+pooling
                configs.append(config)
                variant_cnt += 1
            else:
                metric_rslt = line.split(':')
                metric = metric_rslt[0].strip(' ')
                if variant_cnt <= 1:
                    metrics.append(metric)
                if metric not in y_avg.keys():
                    y_avg[metric] = []
                    y_std[metric] = []
                avg_std = metric_rslt[1].split('+/-')
                y_avg[metric].append(float(avg_std[0].strip(' ')))
                y_std[metric].append(float(avg_std[1].strip('%').strip(' '))/100)
                
print("y_avg:", y_avg)
print("y_std:", y_std)
print('configs:', configs)
print("metrics:", metrics)
# labels = []
# for v in metrics:
#     layer_idx = v.index('layer')
#     labels.append(v[layer_idx+len('layer'):layer_idx+len('layer')+1])
labels = metrics
x_str = []
for config in configs:
    layer_idx = config.index('layer')
    x_str.append(config[layer_idx+len('layer'):layer_idx+len('layer')+1])
print('x_str:', x_str)

alpha=0.9
color_dict = {'red':    '#D62728',
              'green':  '#2CA02C',
              'blue':   '#1F77B4',
              'orange': '#FF7F0E',
              'brown':  '#8C564B',
              'purple': '#9467BD'}
color_list = ['#D62728', '#2CA02C', '#1F77B4', '#FF7F0E', '#9467BD', '#8C564B']
marker_list = ['o', '*', 'd', 's', 'P', 'p', '2', 'X', '|']
linestyle_list = ['solid', 'dotted', 'dashed', 'dashdot']
linewidth = 8
markersize = 18
fontsize = 72
ticks_fontsize = fontsize -9
legend_fontsize = fontsize -9
plt.figure(figsize=(21,15), dpi=80)
std_alpha = 0.1

for i in range(len(metrics)):
    plt.plot(range(len(x_str)), y_avg[metrics[i]], 
        color=color_list[i], marker=marker_list[i], 
        linestyle=linestyle_list[0], label=labels[i], 
        markersize=markersize, alpha=alpha, linewidth=linewidth)
    if args.with_std:
        r1 = list(map(lambda x: x[0]-x[1], zip(y_avg[metrics[i]], y_std[metrics[i]])))#上方差
        r2 = list(map(lambda x: x[0]+x[1], zip(y_avg[metrics[i]], y_std[metrics[i]])))#下方差
        plt.fill_between(range(len(x_str)), r1, r2, color=color_list[i], alpha=std_alpha)
plt.xlabel('layers', fontsize=fontsize, labelpad=-2)
plt.ylabel('Accuracy', fontsize=fontsize)
plt.xticks(range(len(x_str)), x_str, fontsize=ticks_fontsize, rotation=0)
plt.yticks(fontsize=ticks_fontsize)
# plt.legend(loc='upper right', bbox_to_anchor=(1, 2), ncol=2, fontsize=legend_fontsize)
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.73), ncol=2, fontsize=legend_fontsize)
plt.tight_layout()
import os
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
if args.filename == '':
    args.filename = args.datafile.strip('\n').strip('.txt')
plt.savefig(f'{args.outdir}/{args.filename}.{args.filetype}')
plt.close()