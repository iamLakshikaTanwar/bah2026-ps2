PROBLEM STATEMENT 2

Generative AI-Based Cloud Removal and Reconstruction for LISS-IV Satellite Imagery
Description
Persistent cloud cover is a major challenge in optical remote sensing, particularly over tropical and mountainous regions such as the North Eastern Region (NER) of India. Clouds and cloud shadows significantly reduce the usability of optical satellite imagery for applications such as land use–land cover mapping, disaster monitoring, environmental assessment, and infrastructure analysis. LISS-IV imagery provides high spatial resolution data that is valuable for detailed geospatial analysis. However, frequent cloud contamination limits the temporal availability of usable observations. Traditional cloud masking techniques often lead to loss of information and incomplete scene interpretation.

This problem statement focuses on developing a Generative AI-based framework for automated cloud removal and surface reconstruction in LISS-IV imagery. The framework should leverage spatial, spectral, and temporal information to generate cloud-free imagery while preserving fine-scale spatial details and spectral consistency. The proposed solution may explore advanced deep learning approaches such as diffusion models, GANs, transformer-based architectures, or multi-modal fusion techniques using auxiliary data sources such as Sentinel-1 SAR imagery, Sentinel-2 optical imagery, or temporal reference observations.

Objective
Develop a Generative AI-based framework for automated cloud removal in LISS-IV imagery.
Reconstruct cloud-covered regions while preserving spatial structures and spectral characteristics.
Generate visually consistent and analysis-ready cloud-free imagery.
Evaluate the reconstructed outputs using quantitative and qualitative assessment methods.
Develop a scalable workflow for operational cloud removal applications.
Expected Outcomes
Automated cloud-free reconstruction of LISS-IV imagery.
Improved usability of optical satellite data under persistent cloud-cover conditions.
Enhanced spatial and spectral consistency in reconstructed outputs.
Generation of analysis-ready satellite products for downstream geospatial applications.
Development of a prototype framework for operational deployment.
Comparative assessment of different Generative AI architectures for cloud reconstruction.
Dataset Required
Primary Dataset

LISS-IV satellite imagery (cloudy and cloud-free scenes) (Bhoonidhi)
Auxiliary Datasets (Optional publicly available data)
Sentinel-1 SAR imagery
Sentinel-2 optical imagery
Temporal reference imagery
DEM data
Any other datasets
Suggested Tools/Technologies
Programming Frameworks

Python
PyTorch / TensorFlow
Geospatial Tools

GDAL
Rasterio
QGIS
Google Earth Engine (optional)
Supporting Libraries

OpenCV
NumPy
Scikit-image
Albumentations
Expected Solution / Steps to be Followed to Achieve the Objectives
Collection and preprocessing of cloudy and cloud-free LISS-IV imagery.
Preparation of cloud masks and co-registration of auxiliary datasets (optional).
Identification and reuse of suitable pre-trained deep learning or Generative AI models for transfer learning (optional).
Fine-tuning of pre-trained models on LISS-IV imagery for cloud reconstruction tasks (optional).
Integration of temporal and/or multi-sensor information for improved reconstruction quality (optional).
Training, optimization, and validation of the developed deep learning model.
Generation of analysis-ready and cloud-free products.
Documentation and deployment of the developed workflow
