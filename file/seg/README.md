# 算法原理

参考[1802.02611.pdf (arxiv.org)](https://arxiv.org/pdf/1802.02611.pdf)

# 环境安装

参考`pytorch环境安装.pdf`

# 模型训练

```
visdom -port 28849;
python main.py --model deeplabv3plus_mobilenet `
--dataset guazai `
--data_root D:\data\lante_data_argument_datasets `
--enable_vis --vis_port 28849 --gpu_id 0 --year 2012 `
--crop_val --lr 0.01 --crop_size 1024 `
--batch_size 8 --output_stride 16  `
--loss_type ohem_entropy `
--val_interval 30 `
--total_itrs 364400 `
--exp_name train_log_ohem_loss_0119 `
```

# 模型推理

1. 视频推理	

   ```
   python infer_video.py --input "D:\data\1\1_2_3020.MP4"  ` #视频名称
   --dataset guazai `
   --model deeplabv3plus_mobilenet `
   --ckpt D:\cv_project\DeepLabV3Plus-Pytorch\checkpoints\best_deeplabv3plus_mobilenet_guazai_os16_explog_train_log_ohem_loss_0119.pth ` # 权重路径
   --val_batch_size 1 `
   --crop_val `
   --save_val_results_to D:\cv_project\DeepLabV3Plus-Pytorch\z_log\results\01261 # 结果保存路径
   ```


2. 图片推理

   ```
     python infer_simplify_v2.py --input D:\data\lante_data\images  ` # 图片路径
     --dataset guazai `
     --model deeplabv3plus_mobilenet `
     --ckpt D:\cv_project\DeepLabV3Plus-Pytorch\checkpoints\best_deeplabv3plus_mobilenet_guazai_os16_explog_train_log_ohem_loss_0119.pth ` #权重路径
     --val_batch_size 1 `
     --crop_val `
     --save_val_results_to D:\cv_project\DeepLabV3Plus-Pytorch\z_log\results\images #结果保存路径
   ```

  
