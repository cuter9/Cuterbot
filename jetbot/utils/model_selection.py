import pandas as pd
import os

from traitlets import HasTraits, Unicode, List, Bool

from typing import Optional, Tuple
from types import SimpleNamespace

import torch
from torch import Tensor
import torchvision
import torchvision.models as pth_models
# Jetpack used in Jetson nano jetpack does not support torchvision.transforms.v2
if int(torchvision.__version__.split(".")[1]) >= 17:
    from torchvision.transforms.v2 import functional as tt_func, InterpolationMode
    import torchvision.transforms.v2 as tf
    from torchvision import tv_tensors
else:
    from torchvision.transforms import functional as tt_func, InterpolationMode
    import torchvision.transforms as tf

# timm version '0.6.12' for nano
import timm
from timm.data import str_to_interp_mode

HEAD_LIST = ['model_function', 'model_type', 'model_path', 'preprocess_nano_path', "preprocess_path"]
# MODEL_REPO_DIR = os.path.join(os.environ["HOME"], "model_repo")
MODEL_REPO_DIR = os.path.join("/home/cuterbot", "model_repo")
MODEL_REPO_DIR_DOCKER = os.path.join("/workspace", "model_repo")
os.environ['MODEL_REPO_DIR_DOCKER'] = MODEL_REPO_DIR_DOCKER
os.environ['MODEL_REPO_DIR'] = MODEL_REPO_DIR


class ClassifierPreprocessV0(torch.nn.Module):
    # import weights transform function from torchvision v0.19
    def __init__(
            self,
            *,
            crop_size: int = 224,
            resize_size: int = 256,
            mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
            std: Tuple[float, ...] = (0.229, 0.224, 0.225),
            interpolation: InterpolationMode = InterpolationMode.BILINEAR,
            antialias: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.crop_size = [crop_size]
        self.resize_size = [resize_size]
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation
        self.antialias = antialias
        self.tv_version = torchvision.__version__

    def forward(self, img: Tensor) -> Tensor:
        # ref: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py#L39
        # ImageClassification of torchvision is intentionally used for to preprocess image when validating and evaluating
        # use RandomResizedCrop instaed of F.center_crop when training
        img = tt_func.resize(img, self.resize_size, interpolation=self.interpolation, antialias=self.antialias)
        img = tt_func.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = tt_func.pil_to_tensor(img)
        img = tt_func.convert_image_dtype(img, torch.float)
        img = tt_func.normalize(img, mean=self.mean, std=self.std)
        return img

class ClassifierPreprocessV1:
    def __init__(
        self,
        model_config = None,
    ) -> None:
        # self.model_config = model_config
        self.model_config = SimpleNamespace(**model_config)
        self.resize_size = self.model_config.resize_size
        self.crop_size = self.model_config.crop_size
        self.mean = list(self.model_config.mean)
        self.std = list(self.model_config.std)
        self.interpolation = InterpolationMode(self.model_config.interpolation)
        self.antialias = self.model_config.antialias
        print("preprocess configuration: ", model_config)
        # print("resize_size: ", self.resize_size)
        # print("interpolation: ", self.interpolation)

        self.transform_v1 = tf.Compose([tf.Resize(self.crop_size,
                                                  interpolation=self.interpolation,
                                                  antialias=self.antialias),
                                        tf.PILToTensor(),
                                        tf.ConvertImageDtype(torch.float),
                                        tf.Normalize(mean=self.mean, std=self.std)
                                        ])
    def __call__(self, img, is_training=False):
        input_data = img
        output_data = self.transform_v1(input_data)
        img_tf = output_data
        return img_tf

class ClassifierPreprocess:
    def __init__(
        self,
        model_config=None,
    ) -> None:
        self.model_config = model_config
        if hasattr(model_config, "resize_size"):
           self.resize_size = model_config.resize_size
        else:
            self.resize_size = [256, 256]
        if hasattr(model_config, "crop_size"):
            self.crop_size = model_config.crop_size
        else:
            self.crop_size = [224, 224]
        if hasattr(model_config, "mean"):
            self.mean = list(model_config.mean)
        else:
            self.mean = [0.485, 0.456, 0.406]
        if hasattr(model_config, "std"):
            self.std = list(model_config.std)
        else:
            self.std = [0.229, 0.224, 0.225]
        if hasattr(model_config, "interpolation"):
            self.interpolation = model_config.interpolation
        else:
            self.interpolation = InterpolationMode.BILINEAR
        if hasattr(model_config, "antialias"):
            self.antialias = model_config.antialias
        else:
            self.antialias = True
        self.tv_version = torchvision.__version__

        self.transform_train = tf.Compose([tf.RandomHorizontalFlip(p = 0.5),
                                           tf.ColorJitter(0.3, 0.3, 0.3, 0.3),
                                           tf.RandomRotation(15),
                                           tf.Resize(self.resize_size,
                                                   interpolation=self.interpolation,
                                                   antialias=self.antialias),
                                           tf.RandomRotation(15),
                                           tf.RandomCrop(self.crop_size),
                                           tf.PILToTensor(),
                                           tf.ConvertImageDtype(torch.float),
                                           tf.Normalize(mean=self.mean, std=self.std)
                                           ])

        self.transform_val = tf.Compose([tf.Resize(self.crop_size,
                                                   interpolation=self.interpolation,
                                                   antialias=self.antialias),
                                         tf.PILToTensor(),
                                         tf.ConvertImageDtype(torch.float),
                                         tf.Normalize(mean=self.mean, std=self.std)
                                         ])
    @property
    def config(self):
        config = {
            "resize_size": self.resize_size,
            "crop_size":self.crop_size,
            "mean": self.mean,
            "std": self.std,
            "antialias": bool(self.antialias),
            # 內插法模式轉成整數（例如 BILINEAR = 'bilinear'），避免舊版找不到新版的列舉類別
            "interpolation": self.interpolation.value
        }
        return config

    def __call__(self, img, offset=None, is_training=True):
        # x = offset[0]; y = offset[1]
        h, w = img.size
        if offset is None:
            offset_in = [[0.5, 0.5]]
        else:
            offset_in = [[0.5 * h * (offset[0] + 1) , 0.5 * w * (offset[1] + 1)]]

        if is_training:
            offset_kp = tv_tensors.KeyPoints(data=offset_in,
                                             canvas_size=(h, w)
                                             )
            input_data = {"img": img,
                          "poins_of_interest": offset_kp,}
            output_data = self.transform_train(input_data)
            img_tf = output_data["img"]
            offset_tf = (output_data["poins_of_interest"][0] - 0.5 * torch.asarray(self.crop_size) ) / (0.5 * torch.asarray(self.crop_size))
            return img_tf, offset_tf
        else:
            if offset is None:
                input_data = img
                output_data = self.transform_val(input_data)
                img_tf = output_data
                return img_tf
            else:
                offset_kp = tv_tensors.KeyPoints(data=offset_in,
                                                 canvas_size=(h, w)
                                                 )
                input_data = {"img": img,
                              "poins_of_interest": offset_kp,}
                output_data = self.transform_val(input_data)
                img_tf = output_data["img"]
                offset_tf = (output_data["poins_of_interest"][0] - 0.5 * torch.asarray(self.crop_size)) / (
                            0.5 * torch.asarray(self.crop_size))
                return img_tf, offset_tf

def load_pth_model(pth_model_name, weights_cls, pretrained):
    preprocess = None
    model = None
    # for fine-tuning
    if pretrained:
        if weights_cls:
            try:
                weights = getattr(pth_models, weights_cls).DEFAULT
                weights_transforms = weights.transforms()
                model_config = weights_transforms
                model_config.crop_size = [weights_transforms.crop_size[0], weights_transforms.crop_size[0]]
                model_config.resize_size = [weights_transforms.resize_size[0], weights_transforms.resize_size[0]]
                classifier_preprocess = ClassifierPreprocess(model_config)
                # preprocess = [model_config, classifier_preprocess]
                preprocess = [classifier_preprocess.config, classifier_preprocess]
            except AttributeError as err:
                print(f"Attribute Error - {err}! \n"
                      f" Check weights class ( {weights_cls} ) is correct and "
                      f"is available in the torchvision with version {torchvision.__version__}!")

            try:
                model = getattr(pth_models, pth_model_name)(weights=weights, aux_logits=True) \
                    if pth_model_name in ['googlenet', 'inception_v3'] \
                    else getattr(pth_models, pth_model_name)(weights=weights) # for fine-tuning
            except AttributeError as err:
                f"Check {pth_model_name} is available in the torchvision with version {torchvision.__version__}!"

        else:
            try:
                model = getattr(pth_models, pth_model_name)(pretrained=pretrained, aux_logits=True) \
                    if pth_model_name in ['googlenet', 'inception_v3'] \
                    else getattr(pth_models, pth_model_name)(pretrained=pretrained)
                print(f"The  model is loaded from torchvision with version {torchvision.__version__}. \n"
                      "The preprocess of the pretrained weights of torchvision with version >= 0.13 is not applicable!")

            except AttributeError as err:
                print(f"Attribute Error - {err}! \n"
                      f" Check {pth_model_name} is available in the torchvision with version {torchvision.__version__}!")

    else:
        try:
            if weights_cls:
                model = getattr(pth_models, pth_model_name)(weights=None, aux_logits=True) \
                    if pth_model_name in ['googlenet', 'inception_v3'] \
                    else getattr(pth_models, pth_model_name)(weights=None)  # for fine-tuning
            else:
                model = getattr(pth_models, pth_model_name)(pretrained=False, aux_logits=True) \
                    if pth_model_name in ['googlenet', 'inception_v3'] \
                    else getattr(pth_models, pth_model_name)(pretrained=False)

            print(f"The model is loaded from torchvision with version {torchvision.__version__}! \n"
                  "The preprocess of the pretrained weights of torchvision with version >= 0.13 is not applicable!")

        except AttributeError as err:
            print(f"Attribute Error - {err}! \n"
                  f" Check {pth_model_name} is available in the torchvision with version {torchvision.__version__}!.!")

    return model, preprocess

def load_timm_model(timm_model_name, pretrained=True):
    model = None
    preprocess = None
    try:
        model = timm.create_model(timm_model_name, pretrained=pretrained)
    except RuntimeError as err:
        print(f"The model is not available for this timm version {timm.__version__} :  {err}")

    if not pretrained:
        return model, None

    try:
        model_config = timm.get_pretrained_cfg(timm_model_name)
        if hasattr(model_config, "input_size"):
            model_config.resize_size = list(model_config.input_size[1:])
        if hasattr(model_config, "crop_pct"):
            model_config.crop_size = ((torch.asarray(model_config.input_size[1:]) * torch.asarray(model_config.crop_pct)).int()).tolist()
        if hasattr(model_config, "interpolation"):
            model_config.interpolation = str_to_interp_mode(model_config.interpolation)

        classifier_preprocess = ClassifierPreprocess(model_config=model_config)
        preprocess = [classifier_preprocess.config, classifier_preprocess]

    except Exception as err:
        print( f"The configuration of the timm model can not be obtained as required with error: {err}")

    return model, preprocess

def load_model(pth_model_name="resnet18", pretrained=True):
    preprocess = None
    model_type = None
    model = None
    weights_cls = None

    tv = int(torchvision.__version__.split(".")[1])  # torchvision version
    # ----- modify the last layer for classification, and the model used in notebook should be modified too.
    if 'resnet' in pth_model_name:  # ResNet
        model_type = "ResNet"
        if tv >= 13:  # use weights parameter for torchvision with version > 13
            print("torchvision version: %d" % tv)
            weights_cls = pth_model_name.replace("resnet", "ResNet") + "_Weights"

        model, preprocess = load_pth_model(pth_model_name, weights_cls, pretrained)
        if model is not None:
            model.fc = torch.nn.Linear(model.fc.in_features,
                                       2) # for resnet model must add block expansion factor 4

    elif 'mobilenet_v3' in pth_model_name:  # 'mobilenet_v3_large' or  'mobilenet_v3_small'
        model_type = "MobileNet"
        if tv >= 13:  # use weights parameter for torchvision with version > 13
            print("torchvision version: %d" % tv)
            if "small" in pth_model_name:
                weights_cls = "MobileNet_V3_Small_Weights"
            elif "large" in pth_model_name:
                weights_cls = "MobileNet_V3_Large_Weights"
            else:
                assert weights_cls is not None, "Check the use of the name of the torch model!"

        model, preprocess = load_pth_model(pth_model_name, weights_cls, pretrained)
        if model is not None:
            model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features,
                                                  2)  # for mobilenet_v3 model. must add block expansion factor 4
    elif "mobilenetv4" in pth_model_name:
        model_type = "MobileNet"
        # mobilenetv4 is included in timm.models.mobilenetv3
        # ref https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilenetv3.py
        # mobilenetv4_conv_small.e2400_r224_in1k; mobilenetv4_conv_medium.e500_r224_in1k; mobilenetv4_conv_large.e500_r256_in1k
        # model = timm.create_model('mobilenetv4_conv_small', pretrained=pretrained, features_only=True)
        timm_model_name = "mobilenetv4_conv_small"
        if "small" in pth_model_name:
            timm_model_name="mobilenetv4_conv_small"
        elif "medium" in pth_model_name:
            timm_model_name="mobilenetv4_conv_medium"
        elif "large" in pth_model_name:
            timm_model_name="mobilenetv4_conv_large"

        model, preprocess = load_timm_model(timm_model_name, pretrained)
        ## dd = {'device':None , 'dtype': None}
        # model.classifier = Linear(model.head_hidden_size, 2, **dd)
        if model is not None:
            model.classifier = torch.nn.Linear(model.head_hidden_size, 2)

    # for mobilenet_v2 model. must add block expansion factor 4
    elif pth_model_name == 'mobilenet_v2':
        model_type = "MobileNet"
        if tv >= 13:  # use weights parameter for torchvision with version > 13
            print("torchvision version: %d" % tv)
            weights_cls = "MobileNet_V2_Weights"

        model, preprocess = load_pth_model(pth_model_name, weights_cls, pretrained)
        if model is not None:
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features,
                                              2)  # for mobilenet_v2 model. must add block expansion factor 4

    elif pth_model_name == 'vgg11':  # VGGNet
        model_type = "VggNet"
        if tv >= 13:  # use weights parameter for torchvision with version > 13
            print("torchvision version: %d" % tv)
            weights_cls = "VGG11_Weights"

        model, preprocess = load_pth_model(pth_model_name, weights_cls, pretrained)
        if model is not None:
            model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features,
                                              2)  # for VGG model. must add block expansion factor 4

    elif 'efficientnet' in pth_model_name:  # ResNet
        model_type = "EfficientNet"
        if tv >= 13:  # use weights parameter for torchvision with version > 13
            print("torchvision version: %d" % tv)
            if 'efficientnet_b' in pth_model_name:
                weights_cls = pth_model_name.replace("efficientnet_b", "EfficientNet_B") + "_Weights"
            elif 'efficientnet_v2_s' in pth_model_name:
                weights_cls = pth_model_name.replace("efficientnet_v2_s", "EfficientNet_V2_S") + "_Weights"
            elif 'efficientnet_v2_m' in pth_model_name:
                weights_cls = pth_model_name.replace("efficientnet_v2_m", "EfficientNet_V2_M") + "_Weights"
            elif 'efficientnet_v2_l' in pth_model_name:
                weights_cls = pth_model_name.replace("efficientnet_v2_l", "EfficientNet_V2_L") + "_Weights"
            else:
                raise ValueError(f"Unsupported model type {pth_model_name}")

        model, preprocess = load_pth_model(pth_model_name, weights_cls, pretrained)
        if model is not None:
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)  # for efficientnet model
        # model.classifier[0].dropout = torch.nn.Dropout(p=dropout)

    elif pth_model_name == 'inception_v3':  # Inception_v3
        model_type = "InceptionNet"
        if tv >= 13:  # use weights parameter for torchvision with version > 13
            print("torchvision version: %d" % tv)
            weights_cls = "Inception_V3_Weights"

        model, preprocess = load_pth_model(pth_model_name, weights_cls, pretrained)
        # model.dropout = torch.nn.Dropout(p=dropout)
        if model is not None:
            model.fc = torch.nn.Linear(model.fc.in_features, 2)
            if model.aux_logits:
                model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, 2)

    elif pth_model_name == 'googlenet':  # Inception_v3
        model_type = "GoogleNet"
        if tv >= 13:  # use weights parameter for torchvision with version > 13
            print("torchvision version: %d" % tv)
            weights_cls = "GoogLeNet_Weights"

        model, preprocess = load_pth_model(pth_model_name, weights_cls, pretrained)
        if model is not None:
            model.fc = torch.nn.Linear(model.fc.in_features, 2)
            # model.dropout = torch.nn.Dropout(p=dropout)
            if model.aux_logits:
                model.aux1.fc2 = torch.nn.Linear(model.aux1.fc2.in_features, 2)
                model.aux2.fc2 = torch.nn.Linear(model.aux2.fc2.in_features, 2)
            #   model.aux1.dropout = torch.nn.Dropout(p=dropout)
            #   model.aux2.dropout = torch.nn.Dropout(p=dropout)

    elif "densenet" in pth_model_name:  # densenet121, densenet161, densenet169, densenet201
        model_type = "DenseNet"
        if tv >= 13:  # use weights parameter for torchvision with version > 13
            print("torchvision version: %d" % tv)
            weights_cls = pth_model_name.replace("densenet", "DenseNet") + "_Weights"

        model, preprocess = load_pth_model(pth_model_name, weights_cls, pretrained)
        if model is not None:
            model.classifier = torch.nn.Linear(model.classifier.in_features, 2)

    elif "shufflenet_v2" in pth_model_name:  # shufflenet_v2_x1_0 or shufflenet_v2_x0_5
        model_type = "ShuffleNet"
        if tv >= 13:  # use weights parameter for torchvision with version > 13
            print("torchvision version: %d" % tv)
            weights_cls = pth_model_name.replace("shufflenet_v2_x", "ShuffleNet_V2_X") + "_Weights"

        model, preprocess = load_pth_model(pth_model_name, weights_cls, pretrained)
        if model is not None:
            model.fc = torch.nn.Linear(model.fc.in_features, 2)

    elif "mnasnet" in pth_model_name:  # mnasnet1_0 or mnasnet0_5
        model_type = "MnasNet"
        if tv >= 13:  # use weights parameter for torchvision with version > 13
            print("torchvision version: %d" % tv)
            weights_cls = pth_model_name.replace("mnasnet", "MNASNet") + "_Weights"

        model, preprocess = load_pth_model(pth_model_name, weights_cls, pretrained)
        if model is not None:
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)

    elif "vit" in pth_model_name:  #  vit_b_16,  vit_b_32, vit_l_16, vit_l_32, vit_h_14
        # need to pip install flash-attn --no-build-isolation in linux environment only
        # ref: https://stackoverflow.com/questions/78746073/how-to-solve-torch-was-not-compiled-with-flash-attention-warning
        # vit model is not available for jetson nano run in a torch vision version < 0.12
        model_type = "ViTNet"
        # enter the code to convert pytorch 'vit' model so that can be used in Jetbot application.
        if tv >= 13:  # use weights parameter for torchvision with version > 13
            print("torchvision version: %d" % tv)
            weights_cls_lst = list(pth_model_name.replace("vit", "ViT"))
            weights_cls_lst[4] = weights_cls_lst[4].upper()
            weights_cls = ''.join(weights_cls_lst) + "_Weights"

        model, preprocess = load_pth_model(pth_model_name, weights_cls, pretrained)
        if model is not None:
            # model.fc = torch.nn.Linear(model.fc.in_features, 2)
            model.heads[-1] = torch.nn.Linear(model.heads[-1].in_features, 2)
    '''
    else:
        assert (
                model is not None and model_type is not None), \
            f"Check if the model with the model name you set is available in the torchvision package of the version {torchvision.__version__}."

    assert (model is not None), \
        f"Check if the model with the model name you set is available in the torchvision package of the version {torchvision.__version__}."
    '''
    return model, model_type, preprocess


class model_selection(HasTraits):
    model_function = Unicode(default_value='object detection').tag(config=True)
    model_function_list = List(default_value=[]).tag(config=True)
    model_type = Unicode(default_value='SSD').tag(config=True)
    model_type_list = List(default_value=[]).tag(config=True)
    model_path = Unicode(default_value='').tag(config=True)
    model_path_list = List(default_value=[]).tag(config=True)
    selected_model_path = Unicode(default_value='').tag(config=True)
    preprocess_nano_path = Unicode(default_value='').tag(config=True)
    preprocess_path = Unicode(default_value='').tag(config=True)
    is_selected = Bool(default_value=False).tag(config=True)

    def __init__(self, core_library='TensorRT', dir_model_repo=MODEL_REPO_DIR_DOCKER):
        super().__init__()

        self.core_library = core_library
        if self.core_library == 'TensorRT':
            self.df = pd.read_csv(os.path.join(dir_model_repo, "trt_model_tbl.csv"),
                                  header=None, names=HEAD_LIST)
        elif self.core_library == 'Pytorch':
            self.df = pd.read_csv(os.path.join(dir_model_repo, "torch_model_tbl.csv"),
                                  header=None, names=HEAD_LIST)

        for p in self.df.values:
            p[2] = os.path.join(dir_model_repo, p[2].split("/", 1)[1])  # add "workspace" to the path of model
            p[3] = os.path.join(dir_model_repo, p[3].split("/", 1)[1])  # and model preprocess for nano
            p[4] = os.path.join(dir_model_repo, p[4].split("/", 1)[1])  # and model preprocess

        self.model_function_list = list(self.df["model_function"].astype("category").cat.categories)
        self.update_model_type_list()
        # d_mf = self.df[self.df.model_function == self.model_function]   # data frame of given function
        # self.model_type_list = list(d_mf["model_type"].astype("category").cat.categories)
        self.update_model_list()
        # mpl = d_mf[d_mf.model_type == self.model_type].loc[:, ['model_path']].values.tolist()
        # self.model_path_list = np.squeeze(mpl).tolist()
        self.observe(self.update_model, names=['model_function', 'model_type', 'model_path'])
        # self.is_selected = False
        # self.observe(self.selected, names=['is_selected'])

    def update_model_type_list(self):
        mf = self.df[self.df.model_function == self.model_function]  # select the models based on given model function
        # mt = mf[mf.model_type == self.model_type]
        self.model_type_list = list(
            mf["model_type"].astype("category").cat.categories)  # the model types of the given model function
        return self.model_type_list

    def update_model_list(self):
        mf = self.df[self.df.model_function == self.model_function]  # select the models based on given model function
        mt = mf[mf.model_type == self.model_type]  # select the models from the given model type
        mpl = mt.loc[:, ['model_path']].values
        # self.model_path_list = np.squeeze(mpl).tolist()
        self.model_path_list = mpl[:, 0].tolist()
        return self.model_path_list

    def update_model(self, change):
        # print(change)
        if change['name'] == 'model_function':
            self.model_function = change['new']
            self.update_model_type_list()
        if change['name'] == 'model_type':
            self.model_type = change['new']
            self.update_model_list()
        if change['name'] == 'model_path':
            self.model_path = change['new']
            mp = self.df[self.df.model_path == self.model_path]
            # print("model path: ", mp)
            mpp_nano = mp.preprocess_nano_path.tolist()
            self.preprocess_nano_path = mpp_nano[0]
            mpp = mp.preprocess_path.tolist()
            self.preprocess_path = mpp[0]

            # self.selected_model_path = os.path.join(MODEL_REPO_DIR_DOCKER, self.model_path.split("/", 1)[1])
        # print(self.selected_model_path)