from different_arch import vit_arch as vits
from different_arch import deit_arch as deit
from different_arch import cait_arch as cait
from different_arch import xcit_arch as xcit
from different_arch import pit_arch as pit
from different_arch import resnet_arch as resnets
from different_arch import vgg11_arch as vgg11

ARCH_TO_MODEL_PATH = {"vit":"checkpoints/SiT_Small_ImageNet_ViT_student.pth", 
                      "deit":"checkpoints/deit_3_small_224_1k.pth", 
                      "cait":"checkpoints/cait_XXS24_224.pth", 
                      "xcit":"checkpoints/xcit_tiny_12_p16_224.pth", 
                      "pit":"checkpoints/pit_xs_781.pth",
                      "ResNet-18": None, 
                      "WRN-50-2": None, 
                      "vgg11": None}

MODEL_NAME_TO_FN_DICT = {"vit": vits.vit_small_patch16, 
                         "deit": deit.deit_small_patch16_LS, 
                         "cait": cait.cait_XXS24_224, 
                         "xcit": xcit.xcit_tiny_12_p16, 
                         "pit": pit.pit_xs_781, 
                         "ResNet-18": resnets.resnet18, 
                         "WRN-50-2": resnets.wide_resnet50_2, 
                         "vgg11": vgg11.vgg}
