import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--with_std', type=int, default=1)
parser.add_argument('--datafile', type=str, required=True, help='rslt data file')
parser.add_argument('--outdir', type=str, default='./figures')
parser.add_argument('--filename', type=str, default='', help='set the name of the output figure, default for similar to input filename')
parser.add_argument('--filetype', type=str, default='png')
args = parser.parse_args()

# read data to plot
y_avg = {}
y_std = {}
configs = []
metrics = []
with open(f"rslt_data/{args.datafile}", "r") as fin:
    line_cnt = 0
    for line in fin.readlines():
        items = line.strip('\n').split('Avg.')
        config = (items[0].split('_')[1]+'+'+items[0].split('_')[2]).strip(',')
        configs.append(config)
        for i in range(1, len(items)):
            metric_rslt = items[i].split(':')
            metric = metric_rslt[0]
            if line_cnt == 0:
                metrics.append(metric)
            if config not in y_avg.keys():
                y_avg[config] = []
                y_std[config] = []
            avg_std = metric_rslt[1].split('+-')
            y_avg[config].append(float(avg_std[0])/100)
            y_std[config].append(float(avg_std[1][:4])/100)
        line_cnt += 1

print("y_avg:", y_avg)
print("y_std:", y_std)

alpha=0.9
color_dict = {'red':    '#D62728',
              'green':  '#2CA02C',
              'blue':   '#1F77B4',
              'orange': '#FF7F0E',
              'brown':  '#8C564B',
              'purple': '#9467BD'}

# metrics = list(y_avg.keys())
print("metrics:", metrics)
error_kw = {'ecolor' : '0.2', 'capsize' :6 }
alpha=1
linewidth = 8
markersize = 18
fontsize = 60
ticks_fontsize = 60 
legend_fontsize = 60
plt.figure(figsize=(21,15), dpi=80)
xs_str = metrics
x_locs = np.arange(len(xs_str))  # x轴刻度标签位置
width = 0.15 # 柱子宽度
# 6个柱子
if args.with_std == 1:
    bar1 = plt.bar(x_locs-2.5*width, y_avg[configs[0]], yerr=y_std[configs[0]], width=width, error_kw=error_kw, alpha=alpha ,color=color_dict['green'], label=configs[0])
    bar2 = plt.bar(x_locs-1.5*width, y_avg[configs[1]], yerr=y_std[configs[1]], width=width, error_kw=error_kw, alpha=alpha ,color=color_dict['orange'], label=configs[1])
    bar3 = plt.bar(x_locs-0.5*width, y_avg[configs[2]], yerr=y_std[configs[2]], width=width, error_kw=error_kw, alpha=alpha, color=color_dict['blue'], label=configs[2])
    bar4 = plt.bar(x_locs+0.5*width, y_avg[configs[3]], yerr=y_std[configs[3]], width=width, error_kw=error_kw, alpha=alpha, color=color_dict['red'], label=configs[3])
    bar5 = plt.bar(x_locs+1.5*width, y_avg[configs[4]], yerr=y_std[configs[4]], width=width, error_kw=error_kw, alpha=alpha, color=color_dict['purple'], label=configs[4])
    bar6 = plt.bar(x_locs+2.5*width, y_avg[configs[5]], yerr=y_std[configs[5]], width=width, error_kw=error_kw, alpha=alpha, color=color_dict['brown'], label=configs[5])
else:
    bar1 = plt.bar(x_locs-1.5*width, y_avg[configs[0]], width=width, alpha=alpha ,color=color_dict['green'], label=configs[0])
    bar2 = plt.bar(x_locs-0.5*width, y_avg[configs[1]], width=width, alpha=alpha ,color=color_dict['orange'], label=configs[1])
    bar3 = plt.bar(x_locs+0.5*width, y_avg[configs[2]], width=width, alpha=alpha, color=color_dict['blue'], label=configs[2])
    bar4 = plt.bar(x_locs+1.5*width, y_avg[configs[3]], width=width, alpha=alpha, color=color_dict['red'], label=configs[3])
    bar5 = plt.bar(x_locs+1.5*width, y_avg[configs[4]], width=width, alpha=alpha, color=color_dict['purple'], label=configs[4])
    bar6 = plt.bar(x_locs+2.5*width, y_avg[configs[5]], width=width, alpha=alpha, color=color_dict['brown'], label=configs[5])

plt.xticks(x_locs, labels=xs_str, fontsize=ticks_fontsize, rotation=45)
plt.yticks(fontsize=ticks_fontsize)
# plt.ylim(90,100)
plt.ylim(0.85,1.10)
figure_title = args.datafile.split('.')[0]
plt.title(f'{figure_title}', fontsize=fontsize)
plt.xlabel(f'metrics', fontsize=fontsize)
plt.ylabel(f'result', fontsize=fontsize)
a=plt.legend([bar1, bar2, bar3],[configs[0], configs[1], configs[2]], loc="upper right", bbox_to_anchor=(1.02,1.03), fontsize=legend_fontsize)
plt.legend([bar4, bar5, bar6],[configs[3], configs[4], configs[5]],loc="upper left", bbox_to_anchor=(0.00,1.03), fontsize=legend_fontsize)
plt.gca().add_artist(a)
# plt.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=2, fontsize=legend_fontsize)
plt.tight_layout()
import os
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
if args.filename == '':
    args.filename = args.datafile.strip('\n').strip('.txt')
plt.savefig(f'{args.outdir}/{args.filename}.{args.filetype}')
plt.close()