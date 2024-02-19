
from PyQt6.QtGui import QImage, QPixmap
import numpy as np

FMT2CHN_MAP = {
    QImage.Format.Format_RGB32: 4,
    QImage.Format.Format_ARGB32: 4,
    QImage.Format.Format_RGB888: 3,
}

CHN2FMT_MAP = {
    4: QImage.Format.Format_ARGB32,
    3: QImage.Format.Format_RGB888,
}

def formatToChannelNumber(fmt: QImage.Format) -> int:
    return FMT2CHN_MAP[fmt]

def channelNumberToFormat(chn: int) -> QImage.Format:
    return CHN2FMT_MAP[chn]

def convertQImageToMat(origImg: QImage, output_dims: int = 3) -> np.ndarray: 
    '''  Converts a QImage into an opencv MAT format,
     cancels out alpha channel '''
    if output_dims not in [3, 4]:
        raise RuntimeError(f'only RGBA (dims=4) and RGB (dims=3) are supported,: {output_dims}')
    img = origImg.copy().convertToFormat(QImage.Format.Format_RGB32)
    w, h = img.width(), img.height()
    ptr = img.bits()
    ptr.setsize(w * h * 4)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)
    arr = np.copy(arr)
    if output_dims == 4:
        return arr
    elif output_dims == 3:
        return arr[:,:,:3]

def convertMatToQImage(im: np.ndarray) -> QImage:
    h, w, chn = np.shape(im)
    if chn not in [3, 4]:
        raise RuntimeError(f'only RGBA (dims=4) and RGB (dims=3) are supported,: {chn}')
    bytesPerLine = chn * w
    return QImage(im.data, w, h, bytesPerLine, channelNumberToFormat(chn))
