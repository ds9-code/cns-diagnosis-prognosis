# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install required python packages

!pip install SimpleITK
!pip install torchmetrics
!pip install monai

# Commented out IPython magic to ensure Python compatibility.
# Import required python libraries
# These are available libraries and were not coded by the author

# %config InlineBackend.figure_format = 'retina'
# %matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
import tqdm
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import shutil
import datetime
from torchmetrics import Accuracy
from torchmetrics import ConfusionMatrix
import torchmetrics.functional as metric_func
import monai
from sklearn.metrics import roc_curve
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, EnsureType, NormalizeIntensity, RandCoarseShuffle, RandGaussianNoise, RandSpatialCrop
logger = logging.getLogger("iio").setLevel(logging.WARNING)

# Set up paths and image classes

_HF_PATH_TRAIN = "/content/drive/MyDrive/REMBRANDT/Data/Train"
_HF_PATH_VAL = "/content/drive/MyDrive/REMBRANDT/Data/Val"

_HF_CLINICAL_DATA_PATH = "/content/drive/MyDrive/REMBRANDT/rembrandt_clean_hf_series.csv"
_HF_PREFIX = ['axial diffusion',
 'axial t1post gd',
 'axial diff.',
 'axial t1',
 'sag t1',
 'coronal t1 post gd',
 'cor t1 post gad.',
 'axial flair',
 'axial t1 si localizer',
 'axial  fse',
 'na',
 'spectro t1 localizer',
 'sag loc',
 'sag local',
 'axial t1 pre gd',
 'axial perfusion',
 'axial t1 gd',
 'axial t1 post gd',
 'axial fse',
 'axial mpgre',
 'axial mpgr']

_DISEASE_MAPPING = {
    "ASTROCYTOMA":0,
    "GBM":1,
    "OLIGODENDROGLIOMA":2,    
}

_KPS = {
    "80":0,
    "90":1,
    "100":2
}

_GRADE = {
    "II":0,
    "III":1,
    "IV":2
}

# Set up Deep Learning training hyperparameters

cfg = {
    'batch_size':4,
    'lr': 0.0001,
    'epoch': 50,
    'weight_decay':0.001,
    'num_classes': 3,
    'spatial_dims': 3,
    'n_input_channels': 1,
    'device': 'cpu'
}

if torch.cuda.is_available():
     cfg['device'] = 'cuda'

# Function to read all DICOM image slices in a specified folder
def readDicom(path):
    reader = sitk.ImageSeriesReader()
    dir_path = path
    try:
        dicom_names = reader.GetGDCMSeriesFileNames(dir_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
    except:
        print(f"Image not found for patient {dir_path}")
        return
    #Convert to numpy
    np_image = sitk.GetArrayFromImage(image)
    np_image = np_image.astype(np.float32)
    return np_image

# Function to return a list of all the patients in the study
def getPatients(main_dir):
    patients = os.listdir(main_dir)
    return patients

# Function to return a list of all the tests undergone by patients in the study
def getTests(patient_dir):
    patientTests = os.listdir(patient_dir)
    return patientTests

#
def getDicomSamples(test_path):
    dicomSamples = os.listdir(test_path)
    return dicomSamples

def getHF(hf_dir):
    patient_dict = {}
    hf_patients = getPatients(hf_dir)
    #HF patients have only one test
    for patient in hf_patients:
        patient_path = os.path.join(hf_dir, patient)
        tests = getTests(patient_path)
        for test in tests:
            test_path = os.path.join(patient_path, test)
            patient_dict[patient+"__"+test] = getDicomSamples(test_path)
    return patient_dict

# Function to display images for visual inspection
def imshow(img, slice):
    plt.imshow(img[slice], cmap='bone')
    return

# Given a patient ID, the tests and the directory, return the patient's images
def getPatientImage(main_dir, patient_dict, patient, test, prefix):
    """
    Return the first dicom image matching the patient, the test and the prefix
    
    Args:
        main_dir: dataset directory
        patient_dict: dict containing patients, tests and their images of format {patient__test: [dicom1, dicom2, ... , dicomN]}
        patient: patient code we wont to extract the image
        test: the test from which we want to get the image
        prefix: the prefix that determines which image we read
    """
    keys = patient_dict.keys()
    for key in keys:
        patient_test = key.split("__")
        if patient == patient_test[0] and test == patient_test[1]:
            patien_dir = os.path.join(main_dir, patient)
            test_dir = os.path.join(patien_dir, test)
            dicom_list = patient_dict[key]
            print("IMG_LIST: ", dicom_list)
            for dicom in dicom_list:
                print(prefix)
                if prefix.lower() in dicom or prefix.upper() in dicom:
                    dicom_path = os.path.join(test_dir, dicom)
                    img = readDicom(path=dicom_path)
                    return img

# Get all patient IDs from the patient dictionary
def getAllPrefixes(patient_dict):
    prefixes = []
    keys = patient_dict.keys()
    for key in keys:
        dicom_list = patient_dict[key]
        for dicom in dicom_list:
            prefix = dicom.split("-")[1]
            prefixes.append(prefix.lower())
    return list(set(prefixes))

# Get duplicate patient IDs
def getCommonPrefixes(patient_dict, prefixes):
    prefix_dict = {}
    for prefix in prefixes:
        prefix_dict[prefix] = 0
        for key in patient_dict.keys():
            dicom_list = patient_dict[key]
            for dicom in dicom_list:
                if prefix in dicom.lower():
                    prefix_dict[prefix] += 1
                    break
    return prefix_dict

# Delete patient IDs without KPS score
def deleteKpsPatients(main_dir, clinical_file, patient_dict):
    """
    Will delete folders of patients without KPS
    NOTE: deletes the folder
    """
    patients_img = getPatients(main_dir)
    df_patients = pd.read_csv(clinical_file)
    patients_kps = df_patients.Sample.values
    for patient in patients_img:
        if patient not in patients_kps:
          for patient_test in patient_dict.keys():
            patient_d = patient_test.split("__")[0]
            if patient == patient_d:
              del patient_dict[patient_test]

    return patient_dict

# Delete patient IDs with missing data (NA's)
def deleteNaPatients(patient_dict):
    """
    Removes patients with undefined images
    """
    for key in patient_dict.keys():
        img_lis = patient_dict[key]
        for img in img_lis:
            if "NA ".lower() in img.lower():
                del patient_dict[key]
                break
    return patient_dict

# Clean up list of patients to remove missing and incorrect data

patient_dict = getHF(_HF_PATH_TRAIN)
patient_dict = deleteNaPatients(patient_dict)
patient_dict = deleteKpsPatients(main_dir=_HF_PATH_TRAIN, clinical_file=_HF_CLINICAL_DATA_PATH,patient_dict=patient_dict)

# Create a DataReader class

class DataReader(Dataset):
    """
    Parameters:
    ==========
         data_dir: diectory of data
         mri_type: MRI type i.e. axial, coronal or sagital 
         labels_file: file of diagnosis information
         mode: running modality
         transform: transformations on data
    """
    def __init__(self, data_dir, mri_type, labels_file, mode='train', transform=None):
    
        self.data_dir = data_dir
        self.labels_file = labels_file
        self.mode=mode
        self.prefix = mri_type
        self.val_transform = Compose([NormalizeIntensity(), Resize((64, 128, 128), size_mode='all'), EnsureType()])
        self.train_transform = Compose([NormalizeIntensity(), RandRotate90(), RandGaussianNoise(), Resize((64, 128, 128), size_mode='all'), EnsureType()])
        self.patient_test_dict = getHF(self.data_dir)
        self.patient_kps_dict, self.patient_disease_dict, self.patient_grade_dict = self.getLabels() 

    def getLabels(self):
        df_clinical = pd.read_csv(self.labels_file)
        patient_kps_dict = pd.Series(df_clinical['Karnofsky'].values, index=df_clinical.Sample).to_dict()
        patient_disease_dict = pd.Series(df_clinical['Disease'].values, index=df_clinical.Sample).to_dict()
        patient_grade_dict = pd.Series(df_clinical['Grade'].values, index=df_clinical.Sample).to_dict()
        return patient_kps_dict, patient_disease_dict, patient_grade_dict
    
    def getPatientImage(self, idx):
        """
        Return the first dicom image matching the patient, the test and the prefix

        Args:
            main_dir: dataset directory
            patient_dict: dict containing patients, tests and their images of format {patient__test: [dicom1, dicom2, ... , dicomN]}
            patient: patient code we wont to extract the image
            test: the test from which we want to get the image
            prefix: the prefix that determines which image we read
        """
        keys = list(self.patient_test_dict.keys())
        sample = keys[idx]
        patient = sample.split("__")[0]
        test = sample.split("__")[1]
        patien_dir = os.path.join(self.data_dir, patient)
        test_dir = os.path.join(patien_dir, test)
        dicom_list = self.patient_test_dict[sample]
        for dicom in dicom_list:
            if self.prefix.lower() in dicom or self.prefix.upper() in dicom:
                dicom_path = os.path.join(test_dir, dicom)
                img = readDicom(path=dicom_path)
                return img, patient
        dicom_path = os.path.join(test_dir, dicom_list[0])
        img = readDicom(path=dicom_path)
        # if not img.all():
        #   print(dicom_path)
        return img, patient
    
    def __len__(self):
        return len(self.patient_test_dict.keys())
    
    def __getitem__(self, idx):
        np_image, patient = self.getPatientImage(idx)
        print(patient)
        np_image = np.resize(np_image, [1,np_image.shape[1], np_image.shape[2], np_image.shape[0]])
        
        # Get tumor type (label) associated with patient
        label = self.patient_disease_dict[patient]
        label = _DISEASE_MAPPING[label.strip()]

        # Get KPS score associated with patient
        kps = int(self.patient_kps_dict[patient])/100

        # Get tumor grade associated with patient
        grade = self.patient_grade_dict[patient]
        grade = _GRADE[grade.strip()]
        
        if self.mode == "train":
            np_image = self.train_transform(np_image)
        else:
            np_image = self.val_transform(np_image)     
        return np_image, label, kps, grade

# Read and load training data from the Google Drive folder
train_reader = DataReader(data_dir=_HF_PATH_TRAIN, mri_type='axial flair', mode='val', labels_file=_HF_CLINICAL_DATA_PATH)
train_loader = DataLoader(train_reader, batch_size=cfg['batch_size'], shuffle=False, num_workers=0, drop_last=False)

# Read and load validation data from the Google Drive folder
val_reader = DataReader(data_dir=_HF_PATH_VAL, mri_type='axial flair', mode='val', labels_file=_HF_CLINICAL_DATA_PATH)
val_loader = DataLoader(val_reader, batch_size=cfg['batch_size'], shuffle=False, num_workers=0, drop_last=False)

# Create the Resnet network architecture class

class ResNet(torch.nn.Module):
    def __init__(self,num_classes=3, spatial_dims=3, n_input_channels=1):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.spatial_dims = spatial_dims
        self.n_input_channels = n_input_channels

        pretrained = torch.load("/content/drive/MyDrive/REMBRANDT/resnet_18_23dataset.pth")

        pretrained['state_dict'] = {k.replace('module.', ''):v for k, v in pretrained['state_dict'].items()}
        self.net=monai.networks.nets.resnet18(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=2)

        self.net.fc = torch.nn.Linear(in_features=512, out_features=256, bias=True)
        self.cls_fc = torch.nn.Linear(in_features=256, out_features=self.num_classes, bias=True)
        self.reg_fc = torch.nn.Linear(in_features=256, out_features=1, bias=True)
        self.grade_fc = torch.nn.Linear(in_features=256, out_features=3, bias=True)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()
        self.act3 = torch.nn.ReLU()
        
    def forward(self, x):
        embeddings = self.net(x)
        x_cls = self.cls_fc(self.act1(embeddings))
        x_reg = self.reg_fc(self.act2(embeddings))
        x_grade = self.grade_fc(self.act3(embeddings))
        
        return x_cls, x_reg, x_grade

# Construct model with configured parameters
model = ResNet(num_classes=cfg['num_classes'], spatial_dims=cfg['spatial_dims'], n_input_channels=cfg['n_input_channels']).to(cfg['device'])

# Add class weight to handle imbalanced samples = 1-(number of samples of a class/total number of samples)
weights = [0.452830189, 0.735849057, 0.811320755]
class_weights = torch.FloatTensor(weights).cuda()

# Add weights for grade classification = 1-(number of samples of a class/total number of samples)
grade_weights = [0.58490566, 0.679245283, 0.735849057]
grade_weights = torch.FloatTensor(grade_weights).cuda()

# Add weight parameter to CELoss criteria
criterion_1 = torch.nn.CrossEntropyLoss(weight = class_weights)
criterion_2 = torch.nn.MSELoss()
criterion_3 = torch.nn.CrossEntropyLoss(weight = grade_weights)

# Using adaptive weight decay optimizer (AdamW) for better results
optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

# Define model training function
def train(model, train_loader, criterion_1, criterion_2, criterion_3, optimizer, num_epochs=10, val_loader=None, device='cpu'):

    tloss = []
    vloss = []
    ce_loss = []
    mse_loss = []

    for epoch in range(num_epochs):
        print("-"*100)
        now = datetime.datetime.now()
        print(now)
        print(f"epoch {epoch + 1}/{num_epochs}")
        output_labels = torch.empty([0]).to(cfg['device'])
        gt_labels = torch.empty([0]).to(cfg['device'])

        output_grade_ctr = torch.empty([0]).to(cfg['device'])
        gt_grade_ctr = torch.empty([0]).to(cfg['device'])

        train_loss = 0
        val_loss = 0
        train_mse = 0
        train_ce = 0

        train_grade = 0
        val_grade = 0
        
        for i, (data) in enumerate(tqdm.tqdm(train_loader)):

            img, label, kps, grade = data[0].float().to(device), data[1].float().to(device), data[2].float().to(device), data[3].float().to(device)
            optimizer.zero_grad()
            cls_output, reg_output, grade_output = model(img)
            loss_1 = criterion_1(cls_output, label.long())
            loss_2 = criterion_2(reg_output, kps)
            loss_3 = criterion_3(grade_output, grade.long())
            loss = loss_1 + loss_2 + loss_3

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_ce += loss_1.item()
            train_mse += loss_2.item()

            train_grade += loss_3.item()
            
            cls_output = cls_output.argmax(dim=1, keepdim=True).reshape(-1,).detach()
            output_labels = torch.cat((output_labels, cls_output), 0)
            gt_labels = torch.cat((gt_labels, label), 0)

            grade_output = grade_output.argmax(dim=1, keepdim=True).reshape(-1,).detach()
            output_grade_ctr = torch.cat((output_grade_ctr, grade_output), 0)
            gt_grade_ctr = torch.cat((gt_grade_ctr, grade), 0)

        train_loss/=train_loader.__len__()
        train_ce/=train_loader.__len__()
        train_mse/=train_loader.__len__()
        train_grade /= train_loader.__len__()

        # Save training loss history
        tloss.append(train_loss)
        
        # Print all performance metrics after each epoch
        print(f"Train loss {train_loss} = CE {train_ce} + MSE {train_mse} + Grade CE {train_grade}")
        print("Accuracy: ", metric_func.accuracy('multiclass', output_labels.int(), gt_labels.int()))
        precision_recall = metric_func.retrieval_precision_recall_curve(preds=output_labels.int(), target=gt_labels.int())
        print(f"Training precision: {precision_recall[0]}  Recall: {precision_recall[1]}")

        if val_loader:
            with torch.no_grad():
                val_output_labels = torch.empty([0]).to(cfg['device'])
                val_gt_labels = torch.empty([0]).to(cfg['device'])
                val_ce = 0
                val_mse = 0
                #9/10
                val_grade = 0

                val_output_grade_ctr = torch.empty([0]).to(cfg['device'])
                val_gt_grade_ctr = torch.empty([0]).to(cfg['device'])                

                for i, (data) in enumerate(val_loader):
                    image, label, kps, grade = data[0].float().to(device), data[1].to(device), data[2].float().to(device), data[3].float().to(device)
                    cls_output, kps_output, grade_output = model(image)
                    loss_1 = criterion_1(cls_output, label.long())
                    loss_2 = criterion_2(kps_output, kps)
                    loss_3 = criterion_3(grade_output, label.long())
                    loss = loss_1 + loss_2 + loss_3
                    
                    val_loss += loss.item()
                    val_ce += loss_1.item()
                    val_mse += loss_2.item()
                    val_grade += loss_3.item()
                    
                    cls_output = cls_output.argmax(dim=1, keepdim=True).reshape(-1,).detach()
                    val_output_labels = torch.cat((val_output_labels, cls_output), 0)
                    val_gt_labels = torch.cat((val_gt_labels, label), 0)

                    grade_output = grade_output.argmax(dim=1, keepdim=True).reshape(-1,).detach()
                    val_output_grade_ctr = torch.cat((val_output_grade_ctr, grade_output), 0)
                    gt_grade_ctr = torch.cat((gt_grade_ctr, grade), 0)                    

                val_loss/=val_loader.__len__()
                val_ce/=val_loader.__len__()
                val_mse/=val_loader.__len__()
                val_grade /= val_loader.__len__()
                
                # Save validation loss history
                vloss.append(val_loss)
                ce_loss.append(val_ce)
                mse_loss.append(val_mse)

                # Print all the performance metrics after each epoch
                print(f"Val loss {val_loss} = CE {val_ce} + MSE {val_mse} + Grade CE {val_grade}")
                print(f"Val loss {val_loss} = CE {val_ce} + MSE {val_mse}")
                print("Val_accuracy: ", metric_func.accuracy('multiclass', val_output_labels.int(), val_gt_labels.int()))
                confmat = ConfusionMatrix(num_classes=3, multilabel=True).to(cfg['device'])
                confmat(preds=val_output_labels.int(), target=val_gt_labels.int())
                val_precision_recall = metric_func.retrieval_precision_recall_curve(preds=val_output_labels.int(), target=val_gt_labels.int())
                print(f"Val precision: {val_precision_recall[0]}  Val recall: {val_precision_recall[1]}")
                print(confmat)

    # Return losses for charting
    return tloss, vloss, ce_loss, mse_loss

# List number of model parameters
print(sum(p.numel() for p in model.parameters()))

# List number of parameters by module
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
count_parameters(model)

# Run the training using configured hyperparameters and save the loss values for charting

tloss, vloss, ce_loss, mse_loss = train(model=model, train_loader=train_loader, criterion_1=criterion_1, criterion_2=criterion_2, criterion_3=criterion_3, optimizer=optimizer, num_epochs=cfg['epoch'], val_loader=val_loader, device=cfg['device'])

# Generate loss plot to show model's training losses
import matplotlib.pyplot as plt

plt.plot(tloss, label="Training Loss")
plt.plot(vloss, label="Validation Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.title("REMBRANDT - Training vs Validation Loss by Epoch")
plt.legend(loc="upper right")
plt.show()

# Define a model testing function that takes an unseen image and feeds it to the model to get the class predictions

def test(img_path, model):
    transform = Compose([NormalizeIntensity(), Resize((64, 128, 128), size_mode='all'), EnsureType()])
    np_img = readDicom(path=img_path)
    if np_img is None:
        raise ValueError("Input image is not recognized, check the path")
        return
    np_img = np.resize(np_img, [1,np_img.shape[1], np_img.shape[2], np_img.shape[0]])
    image_in = transform(np_img)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model_in = model.to(device)
    image_in = image_in.to(device)
    image_in = image_in.reshape((1, image_in.shape[0], image_in.shape[1], image_in.shape[2], image_in.shape[3]))

    out_label, out_kps, out_grade = model(image_in)
    print(f"predicted out_label is {out_label}. Predicted kps values is {out_kps}. Predicted grade is {out_grade}.")
    
    out_label = out_label.argmax(dim=1, keepdim=True).reshape(-1,).detach()
    for disease in _DISEASE_MAPPING.keys():
        if _DISEASE_MAPPING[disease] == out_label:
            label = disease

    out_grade = out_grade.argmax(dim=1, keepdim=True).reshape(-1,).detach()
    for gr in _GRADE.keys():
        if _GRADE[gr] == out_grade:
            grade = gr

    out_kps = (out_kps*100)
    kps_values = [80, 90, 100]
    difs = []
    for i in kps_values:
        diff = abs(out_kps[0] - i)
        difs.append(diff)
    min_diff = min(difs)
    index = difs.index(min_diff)
    kps = kps_values[index]
    
    return label, kps, grade

# Test the model by providing a new, unseen MRI folder

img_path = '/content/drive/MyDrive/REMBRANDT/Data/Test/HF1139/02-09-1993-NA-NA-05930/4.000000-AXIAL FLAIR-63610'
label, kps, grade = test(img_path=img_path, model=model)

# Print model class predictions
print(f"Predicted kps is {kps} and predicted disease is {label} and tumor grade is {grade}")