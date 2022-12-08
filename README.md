# 2022AMWD-WebshellDetection

## 提交方案


## 消融实验
`ablation_study/`路径下是报告中的消融实验的源码。
数据是官方给定的train.zip和train.csv, 如在本地运行，需要将zip文件解压后放入对应目录下的tctrain/train/，csv文件放入tctrain即可（和比赛提供的路径一致）。

### Graph Convolution Network
`ablation_study/gcn`: 使用GCN模块进行分类。
* gcn_split1.json和gcn_split2.json分别是按照4:1:1和4:1:50划分的数据集进行实验的参数。
* 进入模块， 运行`python main.py --config "gcn_split1.json" `即可，也可以自定义参数设置的json文件。
* statistic.py用于统计最后的实验结果。

### Graph Attention Network
`ablation_study/gat`: 使用GAT模块进行分类。
* `configs/gat/split1`和`configs/gat/split2`分别是按照4:1:1和4:1:50划分的数据集进行实验的参数配置文件。
* 进入`ablation_study/gat`，运行`nohup ./run.sh > run.log 2>&1 &`
* 运行结果将分别附加输出到`configs/gat/split1/gat_split1_rslt.txt`文件和`configs/gat/split2/gat_split1_rslt.txt`文件中

### Adaptive aggregation with Class-Attentive Diffusion
`ablation_study/adaCAD`: 使用adaCAD模块进行分类。
* `configs/adaCAD/split1`和`configs/adaCAD/split2`分别是按照4:1:1和4:1:50划分的数据集进行实验的参数配置文件。
    * "vary_steps"文件夹下的不同配置文件中仅变化`Kstep_for_AdaCAD`这一参数，取值范围为[0,10]。用于探究不同的Kstep对模型效果的影响
    * "kstep4_beta0.8_embdim100"文件夹下的不同配置文件中变化`jk`和`pooling`两个参数。用于探究不同`JK`和`pooling`的选择对模型效果的影响。
* 进入`ablation_study/adaCAD`，运行`nohup ./run.sh > run.log 2>&1 &`
* 运行结果将分别附加输出到相应配置文件夹下的"...rslt.txt"文件中。

### Graph Isomorphism Network
`ablation_study/gin`: 使用GIN模块进行分类。
* `configs/gin/split1`和`configs/gin/split2`分别是按照4:1:1和4:1:50划分的数据集进行实验的参数配置文件。
* 进入`ablation_study/gin`，
    - 运行`nohup sh run1.sh > GinAblitionJK_Pool.log 2>&1 &` 对JK和Pooling参数进行消融实验
        * 运行结果将分别附加输出到`configs/gin/split1/gin_split1rslt.txt`文件和`configs/gin/split2/rslt.txt`文件中
    - 运行`nohup sh run2.sh > GinAblitionLayers.log 2>&1 &` 对Layer参数进行在split2最后的JK:max和pooling:sum的层数进行消融实验
        * 运行结果将附加输出到`configs/gin/split2/gin_split2_rslt.txt`文件中
        

