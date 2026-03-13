# Automated-cytosolic-microtubule-using-object-detection-ML
This repository presents a prototype deep learning pipeline for automated detection of cytosolic microtubules from fluorescence microscopy images using modern object detection techniques. 

# Overview
Cytosolic microtubules play essential roles in cellular structure, intracellular transport, and mitotic spindle organization. Manual quantification of microtubule structures from microscopy images is time-consuming and subject to user bias.

To address this, this project implements an object detection pipeline based on YOLOv10, enabling automated identification of microtubule structures in fluorescence microscopy images. YOLOv10 is a real-time object detection framework optimized for high detection accuracy and efficient inference, allowing rapid analysis of visual data. 

# Workflow
The workflow integrates microscopy imaging, manual annotation using LabelMe, YOLOv10 model training, and web-based inference deployment through a Gradio interface.

The goal of this project is to demonstrate how modern object detection frameworks can assist in automated quantitative microscopy analysis.

![Xnip2026-03-10_22-09-34](https://github.com/user-attachments/assets/a5ff508f-f564-410a-a57b-9f6b6514fd00)

# Detail Steps
1. Microscopy Image Acquisition
Fluorescence microscopy images containing cytosolic microtubule structures were collected and used as the input dataset for model training. These images contain microtubule filament structures that require automated detection and localization.

2. Data Annotation using LabelMe
Manual annotation was performed using the annotation tool LabelMe. Bounding boxes were drawn around cytosolic microtubule structures to generate labeled training data. LabelMe exports annotations in JSON format, which contains the object class labels and coordinates describing the annotated regions.

<img width="668" height="395" alt="image" src="https://github.com/user-attachments/assets/23175b89-736d-44ac-b95d-b20808d7236f" />


4. Dataset Conversion to YOLO Format
The JSON annotation files were converted into the YOLO detection format to enable training with the YOLOv10 model. The YOLO annotation format represents each detected object using normalized bounding box coordinates.Each image has a corresponding annotation file containing the labeled objects.

5. Model Training using YOLOv10
The converted dataset was used to train an object detection model based on YOLOv10, a real-time end-to-end object detection architecture designed for high detection accuracy and efficient inference. Training and experimentation were performed using Kaggle Notebooks and PyTorch / Ultralytics YOLO framework. The trained model was exported in TorchScript format for deployment.

6. Web-Based Inference Platform (Gradio)
To enable interactive testing, the trained YOLO model was integrated into a web interface using Gradio. The Gradio application allows users to:
	•	Upload microscopy images
	•	Run microtubule detection using the trained model
	•	Visualize bounding box predictions

This interface enables rapid demonstration of the model without requiring local installation of the training environment.

Sample figure (to be further refine):

![Xnip2026-01-30_22-00-14](https://github.com/user-attachments/assets/1c65f98b-791c-4646-bca9-f7ce138da6d5)

# Disclaimer
This repository was a Vibe Coding Project for non-coding specialist (like me).
This project was developed using a combination of open computational platforms and AI-assisted coding tools:
	•	Kaggle Notebooks
	•	Google Colaboratory
	•	Python / PyTorch
	•	Gradio
	•	Annotation tool: LabelMe

Code development and prototyping were partially assisted by AI coding tools including:
	•	ChatGPT
	•	Gemini

These tools were used to assist with debugging, workflow design, and implementation of machine learning pipelines.

# Citation
If you use this software, please cite it as: K.H.,Ooi.(2026).Automated-cytosolic-microtubule-using-object-detection-ML. version: 1.0.0. date-released: 2026-03-11.https://github.com/mansonmyr/Automated-cytosolic-microtubule-using-object-detection-ML

