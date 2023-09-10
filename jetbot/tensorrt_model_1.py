"""tensorrt_model.py

This module implements the TRTModel class.
"""


import ctypes

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit



def _preprocess_trt(img, shape=(300, 300)):
    """Preprocess an image before TRT SSD inferencing."""
    img = cv2.resize(img, shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print("Shape of camera input data : ", img.shape,  "Max. and Min. value", np.amax(img),  ", ", np.amin(img), "; \nCamera input data : ", img)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    # img = img.astype(np.float32)
    img *= (2.0/255.0)
    img -= 1.0
    return img


def _postprocess_trt(output, conf_th, output_layout=7):
    """Postprocess TRT SSD output."""
    # img_h, img_w, _ = img.shape
    detections = []
    all_detections = []
    boxes, confs, clss = [], [], []
    for prefix in range(0, len(output), output_layout):
        # if not output[prefix+1] < 0:
        #    print("The detected class : ", output[prefix+1], "\n ---- one detection ---- \n ", output[prefix : prefix+output_layout], "\n")
        #index = int(output[prefix+0])
        conf = float(output[prefix+2])
        if conf < conf_th and output[prefix+1] <= 0:
            continue
        # else:
            # print("The detected class : ", output[prefix+1], "\n ---- one detection ---- \n ", output[prefix : prefix+output_layout], "\n")

        x1 = float(output[prefix+3])
        y1 = float(output[prefix+4])
        x2 = float(output[prefix+5])
        y2 = float(output[prefix+6])
        cls = int(output[prefix+1])
        boxes.append((x1, y1, x2, y2))
        confs.append(conf)
        clss.append(cls)

        det_dict = dict(label = cls,
                confidence = conf,
                bbox = [x1, y1, x2, y2]
        )
        detections.append(det_dict)

    all_detections.append(detections)

    # return boxes, confs, clss
    return all_detections


class TRTModel(object):
    """TrtSSD class encapsulates things needed to run TRT SSD."""

    def _load_plugins(self):
        if trt.__version__[0] < '7':
            ctypes.CDLL("ssd/libflattenconcat.so")
        trt.init_libnvinfer_plugins(self.trt_logger, '')

    def _load_engine(self):
        TRTbin = 'ssd/TRT_%s.bin' % self.model
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings = \
            [], [], [], [], []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
        return host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings

    def __init__(self, model, input_shape=(300, 300), cuda_ctx=None):
        # def __init__(self, model, cuda_ctx=None):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.model = model
        self.input_shape = input_shape
        self.cuda_ctx = cuda_ctx
        if self.cuda_ctx:
            self.cuda_ctx.push()

        self.trt_logger = trt.Logger(trt.Logger.INFO)
        # self._load_plugins()
        # self.engine = self._load_engine()
        with open(self.model, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
             self.engine = runtime.deserialize_cuda_engine(f.read())

        try:
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            self.host_inputs, self.host_outputs, self.cuda_inputs, self.cuda_outputs, self.bindings = self._allocate_buffers()
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e
        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()

    def __del__(self):
        """Free CUDA memories and context."""
        del self.cuda_outputs
        del self.cuda_inputs
        del self.stream

    def detect(self, img, conf_th=0.3):
        """Detect objects in the input image."""
        img_resized = _preprocess_trt(img, self.input_shape)
        # img_resized = _preprocess_trt(img)
        np.copyto(self.host_inputs[0], img_resized.ravel())

        if self.cuda_ctx:
            self.cuda_ctx.push()
        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(
            batch_size=1,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(
            self.host_outputs[1], self.cuda_outputs[1], self.stream)
        cuda.memcpy_dtoh_async(
            self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()
        if self.cuda_ctx:
            self.cuda_ctx.pop()

        output = self.host_outputs[0]
        # print("----- output of detection results ----- \n", output)
        return _postprocess_trt(output, conf_th)

    def __call__(self, inputs):
        return self.detect(inputs)

    def destroy(self):
        self.runtime.destroy()
        self.logger.destroy()
        self.engine.destroy()
        self.context.destroy()