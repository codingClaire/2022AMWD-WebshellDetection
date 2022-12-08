# 对不同组合方式进行实验
nohup sh run1.sh > GinAblitionJK_Pool.log 2>&1 &
for config in 'config_jk1_pooling1.json' 'config_jk1_pooling2.json' 'config_jk1_pooling3.json' 'config_jk2_pooling1.json' 'config_jk2_pooling2.json' 'config_jk2_pooling3.json'; do 
    python main.py --config 'configs/gin/split1/'${config}
done
