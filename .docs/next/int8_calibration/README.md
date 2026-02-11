● Scripts Implementados   

Ambos scripts están listos: 

# 1. extract_calibration_frames.py

Extrae frames de video/imágenes y los preprocesa para calibración.  

```
# Extraer 200 frames de un video
uv run scripts/int8_vnni/extract_calibration_frames.py \
--source videos/sample.mp4 \
--num-frames 200 \  
--resolution 320
```

```
Output: calibration_data/preprocessed_320/*.npy 
```

# 2. calibrate_int8.py (actualizado)  

Calibra modelos ONNX a INT8 con datos reales.   

```
# Calibrar modelo (busca ONNX en raíz)  

uv run scripts/int8_vnni/calibrate_int8.py \
--model yolo11n-seg \   
--resolution 320

```


```
Output: exports/int8_calibrated/segmentation/yolo11n-seg/...xml 
```

--- 
# Flujo Completo  

```

# 1. Extraer frames de calibración (de video representativo)


uv run scripts/int8_vnni/extract_calibration_frames.py \
--source videos/production_sample.mp4 \ 
--num-frames 300 \  
--resolution 320
```

```
## 2. Verificar que hay ONNX disponibles (del export anterior)   

ls *.onnx   

```


```
## 3. Calibrar segmentación  

uv run scripts/int8_vnni/calibrate_int8.py \
--onnx yolo11n-seg_320.onnx \   
--resolution 320
```


```
## 4. Calibrar pose  

uv run scripts/int8_vnni/calibrate_int8.py \
--onnx yolo11n-pose_320.onnx \  
--resolution 320
```

```
# 5. Usar en pipeline híbrido   

uv run run_luna.py \
--video videos/test.mp4 \   
--seg-model exports/int8_calibrated/segmentation/.../model.xml \
--seg-device CPU \  
--pose-model exports/fp16/.../pose.xml \
--pose-device GPU \ 
--show  
```
