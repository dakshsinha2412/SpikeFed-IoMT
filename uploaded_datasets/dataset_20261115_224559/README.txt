SYNTHETIC CLASS-WISE MEDICAL-STYLE IMAGE DATASET

Folder structure:
- Normal/      20 synthetic chest-radiograph-style images
- Pneumonia/   20 synthetic chest-radiograph-style images

Formats included in each class:
JPG, JPEG, PNG, BMP, TIFF

Intended use:
- Test direct image upload and class-wise dataset handling.
- Test image loading and preprocessing:
  grayscale conversion -> 28x28 resize -> tensor conversion -> normalization.

IMPORTANT LIMITATIONS:
These images are procedurally generated illustrations, NOT real chest X-rays.
The folder names are demonstration labels only. The "Pneumonia" images contain
artificial cloudy visual patches and do not represent clinically validated
pneumonia findings. Do not use this dataset to train, validate, or claim
medical diagnostic performance. Use a properly labelled, ethically sourced
real dataset for meaningful medical image classification.
