import os
import paddlehub as hub
 
# 加载模型
seg = hub.Module(name='deeplabv3p_xception65_humanseg', version="1.0.0")  
 
# 要处理的文件目录
path = './images_input/'

# 获取文件列表
files = [path + i for i in os.listdir(path)]  

# 抠图
results = seg.segmentation(data={'image': files}, 
        output_dir = './humanseg_output/')  
