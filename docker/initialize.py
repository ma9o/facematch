import os
import sys
from shutil import copyfileobj
from zipfile import ZipFile
from urllib.request import urlopen
import mxnet as mx
import tvm
import nnvm

model_name = sys.argv[1]
model_url = sys.argv[2]

prefix = 'model-'+model_name+'/model'
epoch = 0

image_size = (112, 112)
opt_level = 3

tmp_file = 'model.zip'
with urlopen(model_url) as r, open(tmp_file, 'wb') as f:
    copyfileobj(r, f)
    f.close()
    with ZipFile(tmp_file, 'r') as z:
        z.extractall()
    os.remove(tmp_file)

sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

shape_dict = {'data': (1, 3, *image_size)}
target = tvm.target.create("llvm")

nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(sym, arg_params, aux_params)
with nnvm.compiler.build_config(opt_level=opt_level):
    graph, lib, params = nnvm.compiler.build(
        nnvm_sym, target, shape_dict, params=nnvm_params)
lib.export_library("./deploy_lib.so")
with open("./deploy_graph.json", "w") as f:
    f.write(graph.json())
with open("./deploy_param.params", "wb") as f:
    f.write(nnvm.compiler.save_param_dict(params))
