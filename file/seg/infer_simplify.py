from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes,guazai
from torchvision import transforms as T
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
from functools import partial
import post_process

#缩放
def resize(image,expected_size=(1024,1024),mode = 'RGB'):
    if isinstance(image,str):
        image=Image.open(image)
    elif isinstance(image,Image.Image):
        pass
    else:
        print("input must be Image.image or image_path")
    
    
    w,h = image.size
    # 长边缩放到指定，短边填0
    ratio = max(h/expected_size[0],w/expected_size[1])
    
    temp_img = image.resize((int(w/ratio),int(h/ratio)))
    dest_img = Image.new(mode,expected_size)
    dest_img.paste(temp_img,(0,0))
    return dest_img,ratio

# 恢复
def recover(image,ratio,orgin_size = (2048,2448)):
    if isinstance(image,str):
        image=Image.open(image)
    elif isinstance(image,Image.Image):
        pass
    else:
        print("input must be Image.image or image_path")
    
    
    w,h = image.size
    # ratio = max(orgin_size[0]/h,orgin_size[1]/w)
    temp_img = image.resize((int(w*ratio),int(h*ratio)))
    ff = temp_img.crop((0,0,orgin_size[1],orgin_size[0]))
    return ff


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, required=True,
                        help="path to a single image or image directory")
    parser.add_argument("--input_lable", type=str, default='',
                        help="GPU ID")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes','guazai'], help='Name of training set')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default=None,
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--infer_size", type=int, default=360)

    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target
    elif opts.dataset.lower() == 'guazai':
        opts.num_classes = 2
        decode_fn = guazai.decode_target


    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)
    
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        
        # torch.onnx.export(model,torch.rand(1,3,1024,1024),r"D:\cv_project\DeepLabV3Plus\checkpoints\latest_deeplabv3plus_mobilenet_guazai_os16_explog__focal_loss_luce_1024_1024_pad_vflip_clean_0625.onnx",input_names=['input_image'],output_names=['preds'])

        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.crop_val:
        transform = T.Compose([
                # T.Resize(360),
                # T.CenterCrop(opts.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = T.Compose([
                # T.Resize(1080),
                # T.CenterCrop(opts.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
        # 保留3份结果
        print('make_dirs')
        concat_dir = os.path.join(opts.save_val_results_to,'concat')
        color_dir = os.path.join(opts.save_val_results_to,'sinlge_color')
        gray_dir = os.path.join(opts.save_val_results_to,'gray_color')
        os.makedirs(concat_dir,exist_ok=True)
        os.makedirs(color_dir,exist_ok=True)
        os.makedirs(gray_dir,exist_ok=True)
        
       
    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            # print(img_path)
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            orgin_img = Image.open(img_path).convert('RGB')
            h,w = orgin_img.size[1],orgin_img.size[0]
            orgin_size = (orgin_img.size[1],orgin_img.size[0])
            temp_img ,ratio= resize(orgin_img,(640,360))
            img = transform(temp_img).unsqueeze(0) # To tensor of NCHW
            
            img = img.to(device)

            pred = model(img).max(1)[1].cpu().numpy()[0] # HW
            pred_post = pred.copy().astype(np.uint8)
            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)
            colorized_preds1=recover(colorized_preds,ratio,orgin_size)
            
            # 灰度图
            gray = Image.fromarray(pred.astype('uint8'))
            gray1 = recover(gray,ratio,orgin_size)

            # 后处理
            # pred_post = post_process.resize_to_original(pred_post,w,h,1/ratio)
            # # pred_post[]
            # pred_post[0:(h//2),:]=0
            # pred_post[-20:,:]=0
            # temp = Image.fromarray(decode_fn(pred_post).astype('uint8'))
            # temp.show()
            
            # img_copy,apps = post_process.simplify(pred_post)
            # le,ri = post_process.get_useful_points(apps,width=w,height=h)
            # print(le,ri)
            # colorized_preds_1 = Image.fromarray(img_copy)
            # colorized_preds_1,post_process.draw_results(colorized_preds_1,le,ri)
            
            
            concat_save = Image.new('RGB',(w*2+20,h),color=(0,0,0))
            concat_save.paste(orgin_img,(0,0))
            concat_save.paste(colorized_preds1,(w+20,0))

            # 保留3份结果
            if opts.save_val_results_to:
                colorized_preds1.save(os.path.join(color_dir, img_name+'.png'))
                concat_save.save(os.path.join(concat_dir, img_name+'.png')) 
                gray1.save(os.path.join(gray_dir, img_name+'.png'))            

if __name__ == '__main__':
    main()
