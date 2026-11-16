"""
dataset_registry.py
Comprehensive registry for Multi-Modal Medical Imaging Datasets in SpikeFed-IoMT.
Maintains clinical metadata, licensing, modality tags, channel specifications,
and status across 12 distinct medical imaging modalities.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

@dataclass
class ModalityMetadata:
    dataset_id: str
    modality: str
    sub_category: str
    display_name: str
    source_citation: str
    license: str
    num_classes: int
    class_labels: List[str]
    input_channels: int  # 1 for Grayscale, 3 for RGB
    default_resolution: tuple[int, int]
    clinical_task: str
    anatomical_site: str
    data_format: str
    is_installed: bool
    sample_count: Dict[str, int]
    description: str

REGISTRY: Dict[str, ModalityMetadata] = {
    "chest_xray": ModalityMetadata(
        dataset_id="chest_xray",
        modality="X-Ray",
        sub_category="Chest Radiography",
        display_name="Pediatric Chest Radiography (Pneumonia Detection)",
        source_citation="Kermany et al., Cell 2018; MedMNIST PneumoniaMNIST v3.0",
        license="CC BY 4.0",
        num_classes=2,
        class_labels=["Normal", "Pneumonia"],
        input_channels=1,
        default_resolution=(28, 28),
        clinical_task="Binary Classification (Normal vs. Bacterial/Viral Pneumonia)",
        anatomical_site="Thorax / Lungs",
        data_format="PNG / Grayscale Tensor",
        is_installed=True,
        sample_count={"train": 40, "val": 0, "test": 0, "total": 40},
        description="High-contrast projection radiographs capturing pulmonary parenchymal consolidation, opacities, and normal anatomical lung markings."
    ),
    "ultrasound": ModalityMetadata(
        dataset_id="ultrasound",
        modality="Ultrasound",
        sub_category="Breast Sonography",
        display_name="Breast Sonography Tumor Assessment",
        source_citation="Al-Dhabyani et al., Data in Brief 2020; MedMNIST BreastMNIST",
        license="CC BY 4.0",
        num_classes=2,
        class_labels=["Malignant", "Benign/Normal"],
        input_channels=1,
        default_resolution=(28, 28),
        clinical_task="Binary Classification (Malignancy Assessment)",
        anatomical_site="Breast / Mammary Gland",
        data_format="PNG / Grayscale Tensor",
        is_installed=False,
        sample_count={"train": 0, "val": 0, "test": 0, "total": 0},
        description="Acoustic impedance cross-sectional scans highlighting hypoechoic masses, posterior acoustic shadowing, and lesion border irregularity."
    ),
    "oct": ModalityMetadata(
        dataset_id="oct",
        modality="OCT",
        sub_category="Retinal Cross-Sectional Tomography",
        display_name="Retinal Optical Coherence Tomography (OCT)",
        source_citation="Kermany et al., Cell 2018; MedMNIST OCTMNIST",
        license="CC BY 4.0",
        num_classes=4,
        class_labels=["CNV", "DME", "Drusen", "Normal"],
        input_channels=1,
        default_resolution=(28, 28),
        clinical_task="Multi-Class Retinopathy & Maculopathy Diagnosis",
        anatomical_site="Retina / Macula",
        data_format="PNG / Grayscale Tensor",
        is_installed=False,
        sample_count={"train": 0, "val": 0, "test": 0, "total": 0},
        description="Interferometric cross-sectional b-scans depicting choroidal neovascularization, macular edema, drusenoid deposits, and retinal pigment epithelial integrity."
    ),
    "dermatology": ModalityMetadata(
        dataset_id="dermatology",
        modality="Dermatology",
        sub_category="Dermoscopy",
        display_name="Pigmented Skin Lesions (HAM10000)",
        source_citation="Tschandl et al., Scientific Data 2018; MedMNIST DermaMNIST",
        license="CC BY-NC 4.0",
        num_classes=7,
        class_labels=["Melanocytic Nevi", "Melanoma", "Basal Cell Carcinoma", "Actinic Keratoses", "Benign Keratosis", "Dermatofibroma", "Vascular"],
        input_channels=3,
        default_resolution=(28, 28),
        clinical_task="Multi-Class Skin Cancer & Lesion Classification",
        anatomical_site="Skin / Cutaneous",
        data_format="PNG / RGB Tensor",
        is_installed=False,
        sample_count={"train": 0, "val": 0, "test": 0, "total": 0},
        description="High-resolution dermatoscope color imaging displaying atypical pigment networks, streaks, blue-white veils, and vascular structures."
    ),
    "histopathology": ModalityMetadata(
        dataset_id="histopathology",
        modality="Histopathology",
        sub_category="Tissue Microscopy",
        display_name="Colorectal Adenocarcinoma Tissue Microscopy",
        source_citation="Kather et al., PLOS Medicine 2016; MedMNIST PathMNIST",
        license="CC BY 4.0",
        num_classes=9,
        class_labels=["Adipose", "Background", "Debris", "Lymphocytes", "Mucus", "Smooth Muscle", "Normal Colon", "Cancer Epithelium", "Stroma"],
        input_channels=3,
        default_resolution=(28, 28),
        clinical_task="Histological Tissue Architecture Classification",
        anatomical_site="Colorectal Mucosa",
        data_format="PNG / RGB Tensor",
        is_installed=False,
        sample_count={"train": 0, "val": 0, "test": 0, "total": 0},
        description="Hematoxylin and eosin (H&E) stained histopathological tissue sections reflecting nuclear atypia, glandular distortion, and lymphoid infiltration."
    ),
    "fundus": ModalityMetadata(
        dataset_id="fundus",
        modality="Fundus",
        sub_category="Color Fundus Photography",
        display_name="Diabetic Retinopathy Fundus Grading",
        source_citation="DeepDR Development Team; MedMNIST RetinaMNIST",
        license="CC BY 4.0",
        num_classes=5,
        class_labels=["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
        input_channels=3,
        default_resolution=(28, 28),
        clinical_task="Ordinal Severity Grading (Diabetic Retinopathy)",
        anatomical_site="Retina / Posterior Pole",
        data_format="PNG / RGB Tensor",
        is_installed=False,
        sample_count={"train": 0, "val": 0, "test": 0, "total": 0},
        description="Full-color fundus photography illuminating microaneurysms, hemorrhages, hard exudates, cotton-wool spots, and retinal neovascularization."
    ),
    "brain_ct": ModalityMetadata(
        dataset_id="brain_ct",
        modality="CT",
        sub_category="Organ & Tissue Tomography",
        display_name="Abdominal Organ & Tissue Cross-Sectional CT",
        source_citation="Bilic et al., TCIA LiTS Challenge; MedMNIST OrganAMNIST",
        license="CC BY 4.0",
        num_classes=11,
        class_labels=["Bladder", "Femur-L", "Femur-R", "Heart", "Kidney-L", "Kidney-R", "Liver", "Lung-L", "Lung-R", "Pancreas", "Spleen"],
        input_channels=1,
        default_resolution=(28, 28),
        clinical_task="Multi-Organ Identification & Segmentation Mapping",
        anatomical_site="Abdomen / Torso",
        data_format="PNG / Grayscale Hounsfield Tensor",
        is_installed=False,
        sample_count={"train": 0, "val": 0, "test": 0, "total": 0},
        description="Hounsfield Unit attenuation cross-sections distinguishing solid organ parenchymal densities, fluid collections, and bony boundaries."
    ),
    "brain_mri": ModalityMetadata(
        dataset_id="brain_mri",
        modality="MRI",
        sub_category="Neuroimaging MRI",
        display_name="Brain Structural MRI (Glioma / Meningioma)",
        source_citation="Cheng et al., Figshare 2017; TCIA Brain Tumor Archive",
        license="CC BY 4.0",
        num_classes=3,
        class_labels=["Meningioma", "Glioma", "Pituitary Tumor"],
        input_channels=1,
        default_resolution=(28, 28),
        clinical_task="Intracranial Lesion Differential Diagnosis",
        anatomical_site="Brain / Neurocranium",
        data_format="PNG / Grayscale T1-CE Tensor",
        is_installed=False,
        sample_count={"train": 0, "val": 0, "test": 0, "total": 0},
        description="T1-weighted contrast-enhanced magnetic resonance axial slices demonstrating intra-axial vs extra-axial dural-based intracranial neoplasms."
    ),
    "bone_xray": ModalityMetadata(
        dataset_id="bone_xray",
        modality="X-Ray",
        sub_category="Musculoskeletal Radiography",
        display_name="Musculoskeletal Radiograph Abnormality (MURA)",
        source_citation="Rajpurkar et al., Stanford ML Group 2018",
        license="Research Use Only",
        num_classes=2,
        class_labels=["Normal", "Abnormal / Fracture"],
        input_channels=1,
        default_resolution=(28, 28),
        clinical_task="Orthopedic Abnormality and Fracture Detection",
        anatomical_site="Upper Extremity (Finger/Wrist/Forearm/Elbow/Shoulder)",
        data_format="PNG / Grayscale Tensor",
        is_installed=False,
        sample_count={"train": 0, "val": 0, "test": 0, "total": 0},
        description="Orthogonal orthopedic projections revealing cortical step-offs, periosteal reaction, joint effusions, and degenerative bone changes."
    ),
    "dental_xray": ModalityMetadata(
        dataset_id="dental_xray",
        modality="X-Ray",
        sub_category="Dental Panoramic Radiography",
        display_name="Panoramic Dental Radiography (Caries & Bone Loss)",
        source_citation="Tufts University Dental Radiograph Collection 2020",
        license="CC BY-NC 4.0",
        num_classes=2,
        class_labels=["Normal", "Pathology (Caries/Periodontal)"],
        input_channels=1,
        default_resolution=(28, 28),
        clinical_task="Dental Pathology & Periodontal Bone Loss Screening",
        anatomical_site="Maxillofacial / Dentition",
        data_format="PNG / Grayscale Tensor",
        is_installed=False,
        sample_count={"train": 0, "val": 0, "test": 0, "total": 0},
        description="Curvilinear orthopantomograms capturing dental crowns, roots, alveolar bone crest levels, and periapical radiolucencies."
    ),
    "mammography": ModalityMetadata(
        dataset_id="mammography",
        modality="X-Ray",
        sub_category="Screening Mammography",
        display_name="Digital Mammography (CBIS-DDSM)",
        source_citation="Lee et al., Scientific Data 2017; TCIA CBIS-DDSM",
        license="TCIA Data Usage Policy",
        num_classes=2,
        class_labels=["Benign", "Malignant"],
        input_channels=1,
        default_resolution=(28, 28),
        clinical_task="Screening Breast Mass & Microcalcification Triage",
        anatomical_site="Breast (CC & MLO Projections)",
        data_format="PNG / Grayscale High Dynamic Range",
        is_installed=False,
        sample_count={"train": 0, "val": 0, "test": 0, "total": 0},
        description="Cranio-caudal and medio-lateral oblique compression mammograms highlighting pleomorphic microcalcifications and architectural distortion."
    ),
    "lung_ct": ModalityMetadata(
        dataset_id="lung_ct",
        modality="CT",
        sub_category="Thoracic Low-Dose CT",
        display_name="Pulmonary Nodule Screening CT (LIDC-IDRI)",
        source_citation="Armato et al., Medical Physics 2011; TCIA LIDC-IDRI",
        license="TCIA Data Usage Policy",
        num_classes=2,
        class_labels=["Non-Nodule / Benign", "Malignant Suspect Nodule"],
        input_channels=1,
        default_resolution=(28, 28),
        clinical_task="Pulmonary Nodule Malignancy Risk Stratification",
        anatomical_site="Thorax / Lungs",
        data_format="PNG / Grayscale Lung Window Tensor",
        is_installed=False,
        sample_count={"train": 0, "val": 0, "test": 0, "total": 0},
        description="Thin-slice axial chest CT reconstructed in lung parenchymal window (-600 HU center, 1500 HU width) tracking solid and sub-solid pulmonary lesions."
    )
}

def get_registry() -> Dict[str, ModalityMetadata]:
    """Returns the complete dataset registry with updated installation states."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for ds_id, meta in REGISTRY.items():
        ds_path = DATA_DIR / ds_id
        if ds_path.is_dir():
            train_count = sum(1 for _ in (ds_path / "train").rglob("*.*")) if (ds_path / "train").exists() else 0
            val_count = sum(1 for _ in (ds_path / "val").rglob("*.*")) if (ds_path / "val").exists() else 0
            test_count = sum(1 for _ in (ds_path / "test").rglob("*.*")) if (ds_path / "test").exists() else 0
            total = train_count + val_count + test_count
            if total > 0:
                meta.is_installed = True
                meta.sample_count = {"train": train_count, "val": val_count, "test": test_count, "total": total}
    return REGISTRY

def get_modality(dataset_id: str) -> Optional[ModalityMetadata]:
    reg = get_registry()
    return reg.get(dataset_id)

def list_modalities() -> List[Dict[str, Any]]:
    reg = get_registry()
    return [asdict(m) for m in reg.values()]

if __name__ == "__main__":
    print(f"SpikeFed-IoMT Registry initialized with {len(REGISTRY)} modalities.")
    for m in list_modalities():
        status = "INSTALLED" if m["is_installed"] else "REGISTERED"
        print(f"  [{status:11s}] {m['modality']:14s} | {m['display_name']} ({m['num_classes']} classes)")
