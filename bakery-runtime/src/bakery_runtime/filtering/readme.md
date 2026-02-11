#Designing Filtering Sub-Package


Es un patrón que se ve en pipelines de visión en producción (DeepStream, Supervision, etc.): separar el decodificado numérico del modelo de la lógica de "qué me quedo".

La propuesta clave:

FilterPolicy — un dataclass declarativo que captura todas las reglas de filtrado (confianza global, por modelo, por clase, por keypoint). Value object puro, sin estado.
filter_detections() / filter_keypoints() — funciones puras numpy que aplican la policy. Se ejecutan post-NMS, pre-entity construction (el punto más temprano útil en la pipeline).
bakery_runtime.filtering/ — sub-paquete propio, separado de processing/ (que queda raw postprocess).
CLI con listas pareadas — --classes 0 2 + --class-confidence 0.6 0.3, igual para keypoints.


Design Proposal: bakery-runtime.filtering Sub-Package

Context

Currently confidence thresholding is a single scalar baked into postprocess_segmentation  and postprocess_pose. The --classes filter we just added is also inlined. As the pipeline matures, we need multi-level, composable filtering without polluting the postprocess signatures with ever-growing parameter lists.

This maps to a well-known pattern in ML pipeline design — what NVIDIA DeepStream calls secondary inference filtering and what Roboflow Supervision implements via chained sv.Detections operations. The core idea: separate "what the model outputs" from "what the application keeps".


bakery run \
  --confidence 0.25 \                          # global
  --seg-confidence 0.3 \                       # seg override
  --pose-confidence 0.2 \                      # pose override
  --classes 0 2 \                              # whitelist
  --class-confidence 0.6 0.3 \                 # paired with --classes
  --keypoints 9 15 \                           # keypoint indices to threshold
  --keypoint-confidence 0.3 0.4 \              # paired with --keypoints
  --keypoint-min-visible 5                     # drop skeleton if < 5 visible


---

Walkthrough: bakery_runtime.filtering Sub-Package

Summary

Implemented a composable post-inference filtering system as a new sub-package inside bakery-runtime. This replaces the inline classes parameter that was previously added to postprocess_segmentation, and adds multi-level confidence thresholding (global → per-model → per-class → per-keypoint).


CLI Usage Examples

# Class filtering (same as before, but now via FilterPolicy)
bakery run --classes 0
# Per-class confidence
bakery run --classes 0 2 --class-confidence 0.6 0.3
# Per-model confidence
bakery run --seg-confidence 0.4 --pose-confidence 0.15
# Keypoint filtering
bakery run --keypoints 9 15 --keypoint-confidence 0.3 0.4
# Drop low-quality skeletons
bakery run --keypoint-min-visible 5