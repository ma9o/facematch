import rpyc
from scipy.spatial import distance
import mxnet as mx
import face_recognition
import os

rpc_port = 50052
image_size = (112, 112)


def load_img(filename):
    filename = os.path.join(os.path.dirname(globals()["__file__"]), filename)
    try:
        img = mx.image.imread(filename)
    except Exception as e:
        raise RuntimeError('Couldn\'t load '+filename) from e
    face_coord = face_recognition.face_locations(img.asnumpy())
    if(len(face_coord) == 0):
        raise RuntimeError('No faces in '+filename)
    img = mx.image.fixed_crop(img, face_coord[0][3], face_coord[0][0], abs(
        face_coord[0][1]-face_coord[0][3]), abs(face_coord[0][0]-face_coord[0][2]), size=image_size)
    img = img.transpose((2, 0, 1))
    img = img.expand_dims(axis=0)
    img = img.astype('float32')
    return img.asnumpy().ravel()


class FaceMatch:
    def __init__(self, pic_a, pic_b):
        rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
        rpyc.core.protocol.DEFAULT_CONFIG['allow_public_attrs'] = True
        rpyc.core.protocol.DEFAULT_CONFIG['allow_all_attrs'] = True
        try:
            self.rpc = rpyc.connect(
                "localhost", rpc_port, config=rpyc.core.protocol.DEFAULT_CONFIG)
        except Exception as e:
            raise RuntimeError(
                'RPC connection failed. Is the Docker container running?') from e
        self.pic_a = load_img(pic_a)
        self.pic_b = load_img(pic_b)

    def compare(self):
        vec_a = self.rpc.root.evaluate(self.pic_a)
        vec_b = self.rpc.root.evaluate(self.pic_b)
        return 1-distance.cosine(vec_a, vec_b)
