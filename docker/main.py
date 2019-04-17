import rpyc
from rpyc.utils.server import ThreadedServer
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import tvm
from tvm.contrib import graph_runtime


class FaceMatchService(rpyc.Service):
    def __init__(self):
        loaded_json = open("./deploy_graph.json").read()
        loaded_lib = tvm.module.load("./deploy_lib.so")
        loaded_params = bytearray(open("./deploy_param.params", "rb").read())
        self.module = graph_runtime.create(loaded_json, loaded_lib, tvm.cpu())
        self.module.load_params(loaded_params)

    def exposed_evaluate(self, img):
        print('RPC call recieved', img)
        tmp = np.ndarray(img.shape, buffer=np.array(img), dtype='float32')
        self.module.run(data=tvm.ndarray.array(tmp))
        res = tvm.nd.empty((512,))
        self.module.get_output(0, out=res)
        return res.asnumpy()


if __name__ == "__main__":
    rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
    rpyc.core.protocol.DEFAULT_CONFIG['allow_public_attrs'] = True
    rpyc.core.protocol.DEFAULT_CONFIG['allow_all_attrs'] = True
    server = ThreadedServer(FaceMatchService, port=50052,
                            protocol_config=rpyc.core.protocol.DEFAULT_CONFIG)
    server.start()
