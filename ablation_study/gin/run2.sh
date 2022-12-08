# 对不同组合方式进行实验
# nohup sh test.sh > 1204ginablition.log 2>&1 &
# for config in 'config_jk1_pooling1.json' 'config_jk1_pooling2.json' 'config_jk1_pooling3.json' 'config_jk2_pooling1.json' 'config_jk2_pooling2.json' 'config_jk2_pooling3.json'; do 
#     python main.py --config 'configs/gin/split1/'${config}
# done

# 选择jk:max pooling为sum进行实验对层数的消融实验
# nohup sh test.sh > 1207ginLayers1~5.log 2>&1 &
for layer_num in 1 2 3 4 5; do
    python main.py --config 'configs/gin/split2/config_jk2_pooling3.json' --num_layer ${layer_num}
done