#--------------------------------------------#
#   该部分代码用于看网络参数
#--------------------------------------------#
import torch
from thop import clever_format, profile

from nets.detr import DETR

if __name__ == '__main__':
    input_shape         = [800, 800]
    backbone            = "resnet50"
    position_embedding  = 'sine'
    hidden_dim          = 256
    num_classes         = 92
    num_queries         = 100
    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m       = DETR(backbone, position_embedding, hidden_dim, num_classes=num_classes, num_queries=num_queries)
    for i in m.children():
        print(i)
        print('==============================')
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(m.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
