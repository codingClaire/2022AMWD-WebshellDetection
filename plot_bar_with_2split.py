import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--with_std', type=int, default=1)
parser.add_argument('--split1', type=str, required=True, help='split1 rslt data file')
parser.add_argument('--split2', type=str, required=True, help='split2 rslt data file')
parser.add_argument('--outdir', type=str, default='./figures')
parser.add_argument('--filename', type=str, default='', help='set the name of the output figure, default for similar to input filename')
parser.add_argument('--filetype', type=str, default='png')
args = parser.parse_args()

# read data to plot
y_avg = {}
y_std = {}
with open(f"rslt_data/{args.split1}", "r") as fin:
    for line in fin.readlines():
        items = line.strip('\n').split('Avg.')
        for i in range(1, len(items)):
            metric_rslt = items[i].split(':')
            metric = metric_rslt[0]
            if metric not in y_avg.keys():
                y_avg[metric] = []
                y_std[metric] = []
            avg_std = metric_rslt[1].split('+-')
            y_avg[metric].append(float(avg_std[0]))
            y_std[metric].append(float(avg_std[1][:4]))

with open(f"rslt_data/{args.split2}", "r") as fin:
    for line in fin.readlines():
        items = line.strip('\n').split('Avg.')
        for i in range(1, len(items)):
            metric_rslt = items[i].split(':')
            metric = metric_rslt[0]
            if metric not in y_avg.keys():
                y_avg[metric] = []
                y_std[metric] = []
            avg_std = metric_rslt[1].strip('%').split('+-')
            y_avg[metric].append(float(avg_std[0]))
            y_std[metric].append(float(avg_std[1][:4]))

print("y_avg:", y_avg)
print("y_std:", y_std)

alpha=0.9
color_dict = {'red':    '#D62728',
              'green':  '#2CA02C',
              'blue':   '#1F77B4',
              'orange': '#FF7F0E',
              'brown':  '#8C564B',
              'purple': '#9467BD'}

metrics = list(y_avg.keys())
print("metrics:", metrics)
error_kw = {'ecolor' : '0.2', 'capsize' :6 }
alpha=1
linewidth = 8
markersize = 18
fontsize = 60
ticks_fontsize = 60 
legend_fontsize = 60
plt.figure(figsize=(21,15), dpi=80)
xs_str = ['split1', 'split2']
x_locs = np.arange(len(xs_str))  # x轴刻度标签位置
width = 0.15 # 柱子宽度
# 四个柱子
if args.with_std == 1:
    bar1 = plt.bar(x_locs-1.5*width, y_avg[metrics[0]], yerr=y_std[metrics[0]], width=width, error_kw=error_kw, alpha=alpha ,color=color_dict['green'], label=metrics[0])
    bar2 = plt.bar(x_locs-0.5*width, y_avg[metrics[1]], yerr=y_std[metrics[1]], width=width, error_kw=error_kw, alpha=alpha ,color=color_dict['orange'], label=metrics[1])
    bar3 = plt.bar(x_locs+0.5*width, y_avg[metrics[2]], yerr=y_std[metrics[2]], width=width, error_kw=error_kw, alpha=alpha, color=color_dict['blue'], label=metrics[2])
    bar4 = plt.bar(x_locs+1.5*width, y_avg[metrics[3]], yerr=y_std[metrics[3]], width=width, error_kw=error_kw, alpha=alpha, color=color_dict['red'], label=metrics[3])
else:
    bar1 = plt.bar(x_locs-1.5*width, y_avg[metrics[0]], width=width, alpha=alpha ,color=color_dict['green'], label=metrics[0])
    bar2 = plt.bar(x_locs-0.5*width, y_avg[metrics[1]], width=width, alpha=alpha ,color=color_dict['orange'], label=metrics[1])
    bar3 = plt.bar(x_locs+0.5*width, y_avg[metrics[2]], width=width, alpha=alpha, color=color_dict['blue'], label=metrics[2])
    bar4 = plt.bar(x_locs+1.5*width, y_avg[metrics[3]], width=width, alpha=alpha, color=color_dict['red'], label=metrics[3])

plt.xticks(x_locs, labels=xs_str, fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
# plt.ylim(0.53,1.05)
# plt.title(f'{args.dataset.capitalize()}')
plt.xlabel(f'划分比例', fontsize=fontsize)
plt.ylabel(f'评价指标', fontsize=fontsize)
a=plt.legend([bar1, bar2],[metrics[0], metrics[1]], loc="upper right", bbox_to_anchor=(1.02,1.03), fontsize=legend_fontsize)
plt.legend([bar3, bar4],[metrics[2], metrics[3]],loc="upper left", bbox_to_anchor=(0.00,1.03), fontsize=legend_fontsize)
plt.gca().add_artist(a)
# plt.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=2, fontsize=legend_fontsize)
plt.tight_layout()
import os
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
if args.filename == '':
    tmp = args.split1.split('_split1')
    args.filename = tmp[0]+tmp[1].split('.')[0]
plt.savefig(f'{args.outdir}/{args.filename}.{args.filetype}')
plt.close()