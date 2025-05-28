import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine("lightweight_unet_jetson.engine")
context = engine.create_execution_context()

# Alocação de buffers
input_shape = (1, 3, 256, 256)  
output_shape = (1, 1, 256, 256)
d_input = cuda.mem_alloc(np.prod(input_shape) * np.float32().nbytes)
d_output = cuda.mem_alloc(np.prod(output_shape) * np.float32().nbytes)
bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Pré-processamento
    img = cv2.resize(frame, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # CxHxW
    img = np.expand_dims(img, axis=0)  # 1x3x256x256
    img = np.ascontiguousarray(img)

    # Copiar para GPU
    cuda.memcpy_htod_async(d_input, img, stream)

    # Inferência
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Recuperar resultado
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()

    # Pós-processamento
    pred_mask = (output[0, 0] > 0.5).astype(np.uint8) * 255
    pred_mask = cv2.resize(pred_mask, (frame.shape[1], frame.shape[0]))
    overlay = frame.copy()
    overlay[pred_mask > 0] = (0, 255, 255)  # amarelo

 
    cv2.imshow("Overlay", overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

