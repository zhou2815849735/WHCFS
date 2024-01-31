import cv2
from  skimage import morphology
from skimage.morphology import remove_small_holes,remove_small_objects,disk
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def simplify(img, min_hole=300, min_object=300):
    dtype = img.dtype
    assert dtype == np.uint8
    class_ids = np.unique(img)[1:]  # exclude zero value
    # class_ids=[]
    img_copy = img.copy()
    img_copy = np.zeros(img.shape).astype(np.uint8)
    appoxs = []
    for class_id in class_ids:
        if class_id ==1:
        
            tmp_img = img == class_id
            # morphology.binary_closing(tmp_img, footprint=disk(10), out=tmp_img)
            tmp_img=remove_small_holes(tmp_img, min_hole)
            tmp_img=remove_small_objects(tmp_img, min_object)
            
            counters,_= cv2.findContours(np.where(tmp_img,255,0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            for counter in counters:
                epsilon = 0.005*cv2.arcLength(counter,True) 
                appox = cv2.approxPolyDP(counter,epsilon,closed=False).squeeze(1)
                cv2.fillPoly(img_copy,np.array([appox]),int(class_id))
                appoxs.append(appox) 

    return img_copy,appoxs


def get_useful_points(apps,width=640,height=360):
    '''
    apps 为多边形，理想形状为梯形 
    example:[[[1,2],[3,4],[5,6]],[[7,8],[8,9],[9,10]]]
    '''

    points_= []
    for a in apps:
        index = np.where(a[:,1]==(height*2//3))
        points_.extend(list(a[index]))
    points1_ = sorted(points_,key = lambda x: x[0])

    # 分组，左右两组,分组时取下方的值,这一步先保留，调试用
    points2_= []
    # for b in apps:
    #     index2 = np.where(a[:,1]==(height-40))
    #     points2_.extend(list(b[index2]))
    # points2_ = sorted(points2_,key = lambda x: x[0])

    # 设置中点横坐标
    middle = width//2

    left_points = []
    right_points = []
    for ind,i in enumerate(points1_):
        if i[0]-middle<0:
            left_points.append(points1_[ind])
        else:
            right_points.append(points1_[ind])

    # 输出左点和右点
    le = left_points[-1]
    ri = right_points[0]

    return le,ri

def draw_results(color_img,le,ri):
    w,h = color_img.size
    font = ImageFont.truetype(
    font='simhei.ttf',
    size=np.floor(h/10).astype('int32')
) 
    draw = ImageDraw.Draw(color_img)
    draw.ellipse(((le[0]-3,le[1]-3), (le[0]+3,le[1]+3)), fill=(0, 0, 255), outline=(0, 0, 255), width=20)
    draw.ellipse(((ri[0]-3,ri[1]-3), (ri[0]+3,ri[1]+3)), fill=(0, 0, 255), outline=(0, 0, 255), width=20)
    draw.text(list(le),str(le[0]),fill=(255,0,0),font=font)
    draw.text(list(ri),str(ri[0]),fill=(0,255,0),font=font)
    return color_img



def resize_to_original(image_matrix, original_width, original_height, scale_factor):
    # 计算当前图片宽高
    h,w = image_matrix.shape[0:2]
    # 计算裁剪之前的宽高
    
    temp_h = int(h / scale_factor)
    temp_w = int(w / scale_factor)


    # 使用 OpenCV 的 resize 方法进行缩放
    resized_image = cv2.resize(image_matrix, (temp_w, temp_h), interpolation=cv2.INTER_LINEAR)

    # 创建原始大小的空白画布
    original_size_image = np.zeros((original_height, original_width), dtype=np.uint8)

   
    # 将缩放后的图像复制到原始大小的画布的合适位置
    if len(image_matrix.shape)==3:
        original_size_image = resized_image[0:original_height, 0:original_width,:] 
    else:
        original_size_image = resized_image[0:original_height, 0:original_width] 

    return original_size_image

def resize_with_padding(image_matrix, target_width, target_height):
    # 获取原始图像的宽度和高度
    # 创建目标大小的空白画布

    padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    if len(image_matrix.shape)==3:
        original_height, original_width, _ = image_matrix.shape
        padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    else:
        original_height, original_width = image_matrix.shape
        padded_image = np.zeros((target_height, target_width), dtype=np.uint8)


    # 计算缩放比例，以保持长宽比
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height
    scale_factor = min(width_ratio, height_ratio)

    # 计算缩放后的宽度和高度
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # 使用 OpenCV 的 resize 方法进行缩放
    resized_image = cv2.resize(image_matrix, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # 计算填充的位置
    start_x = (target_width - new_width) // 2
    start_y = (target_height - new_height) // 2

    # 将缩放后的图像复制到画布的合适位置
    padded_image[0:new_height, 0:new_width, :] = resized_image

    return padded_image,scale_factor

def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    cmap = cmap/255 if normalized else cmap
    return cmap


def merge(image,mask):
    color_img  = Image.fromarray(voc_cmap()[np.array(mask)])  
    rgb_colorpil = color_img.convert('RGBA')
    rgb_colorpil_array=np.array(rgb_colorpil)
    dest = np.array(image.convert('RGBA'))

    index_mask = np.where(np.array(mask)!=0)
    dest_ = dest.copy()
    dest_[index_mask[0],index_mask[1],:]=dest[index_mask[0],index_mask[1],:]*0.5+rgb_colorpil_array[index_mask[0],index_mask[1],:]*0.5
    blend = Image.fromarray(dest_)
    return blend
