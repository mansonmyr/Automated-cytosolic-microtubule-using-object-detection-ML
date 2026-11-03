# Third-party notices

The current project's original code and documentation use GPL-3.0-only, as stated in [LICENSE](LICENSE). Third-party components retain their own terms. The previous repository tree and its license are preserved under [archive/legacy-before-1024-release/](archive/legacy-before-1024-release/).

| Component | Use | Upstream license and source |
| --- | --- | --- |
| YOLOv10 / Ultralytics | Detector implementation and pretrained-model lineage used to create `microtubule-studio/dist/refined1024.onnx` | AGPL-3.0; [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10), [Ultralytics](https://github.com/ultralytics/ultralytics) |
| ONNX Runtime Web 1.20.1 | Browser inference, loaded from jsDelivr | MIT; [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime) |
| UTIF.js 3.1.0 | TIFF decoding, loaded from jsDelivr | MIT; [photopea/UTIF.js](https://github.com/photopea/UTIF.js) |

The upstream AGPL text is included in [LICENSES/AGPL-3.0.txt](LICENSES/AGPL-3.0.txt). The selected ONNX artifact is distributed subject to applicable upstream model terms; it is not offered as a GPL-only replacement for those terms. No separate commercial license is recorded in the supplied project files.

Training and analysis scripts import external Python packages. Those packages are installed separately and retain their own licenses. Raw research images are not part of this distribution; their reuse permissions are not granted by the software license.
