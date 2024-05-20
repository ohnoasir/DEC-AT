数据集名称更改为VOC2012和clipart

#运行
python lowlight_test.py
#再运行
!python train_net.py \
 --num-gpus 1 \
 --config configs/1.yaml\
 OUTPUT_DIR output/exp_5 


