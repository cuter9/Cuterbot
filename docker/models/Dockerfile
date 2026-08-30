# ARG BASE_IMAGE=jetbot-base:jp44
ARG BASE_IMAGE=${BASE_IMAGE}
FROM ${BASE_IMAGE}

RUN python3 -c "import torchvision; \
                model_alexnet = torchvision.models.alexnet(pretrained=True); \
                model_squeezenet1_1 = torchvision.models.squeezenet1_1(pretrained=True); \
                model_resnet18 = torchvision.models.resnet18(pretrained=True); \
                model_resnet34= torchvision.models.resnet34(pretrained=True)"
