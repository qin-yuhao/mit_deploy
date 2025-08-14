from rknnlite.api import RKNNLite
import numpy as np
RK3576_RKNN_MODEL = '/home/cat/mit_deploy/policy.rknn'

rknn_lite = RKNNLite()
ret = rknn_lite.load_rknn(RK3576_RKNN_MODEL)
if ret != 0:
    print('Load RKNN model failed')
    exit(ret)

zeros = np.ones((1, 45), dtype=np.float32)

ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
if ret != 0:
    print('Init runtime environment failed')
    exit(ret)
outputs = rknn_lite.inference(inputs=[zeros])
if ret != 0:
    print('Inference failed')
    exit(ret)

print(outputs)
