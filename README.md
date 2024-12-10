# Final Modifications in Zero and HWNASBench Libraries

In the final part, we modified the Zero and HWNASBench libraries to achieve our goals. We updated the `main.py` file and the `__init__.py` file in the Measure section of Zero to ensure no conflict with the existing functionality. The modifications are summarized as follows:

### Modifications in `main.py`:
1. Introduced `HW_NAS_Bench_api` to obtain architecture latency and energy consumption on specific hardware.
2. Added a latency-based scoring mechanism and NRS final score calculation functionality.
3. Added a filtering function to filter and store architectures based on specified criteria.
4. Removed irrelevant content and parameters.

### Modifications in `__init__.py`:
1. Refactored the header file.
2. Refactored dynamic loading and lazy loading in the `ZeroShot_test` section.
3. Added an NRS calculation function.


**Details of the design approach can be found in the report.**

---

### **Step P1: Check GPU Availability**

```python
import tensorflow as tf

if tf.config.list_physical_devices('GPU'):
    print("GPU is available!")
else:
    print("No GPU found.")

!mkdir /home/test0
%cd /home/test0
!pwd
```

---

### **Step P2: Clone and Install NASBench**

```bash
!git clone https://github.com/google-research/nasbench /home/test0/nasbench
%cd /home/test0/nasbench
!sed -i 's/import tensorflow as tf/import tensorflow.compat.v1 as tf/g' nasbench/api.py
!sed -i 's/import tensorflow as tf/import tensorflow.compat.v1 as tf/g' nasbench/lib/evaluate.py
!sed -i 's/import tensorflow as tf/import tensorflow.compat.v1 as tf/g' nasbench/lib/training_time.py
%cd /home/test0/nasbench
!pip install -e .
```

---

### **Step P3: Download Datasets**

```bash
# Create necessary directories
!mkdir -p /home/test0/dataset/nasbench/
!mkdir -p /home/test0/dataset/nasbench/NATS/

# Download NASBench-101 dataset
%cd /home/test0/dataset/nasbench/
!wget https://storage.googleapis.com/nasbench/nasbench_full.tfrecord  # NASBench-101

# Download NASBench-201 dataset
!gdown https://drive.google.com/uc?id=16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_  # NASBench-201

# Download NATS dataset
%cd /home/test0/dataset/nasbench/NATS/
!gdown https://drive.google.com/uc?id=1scOMTUwcQhAMa_IMedp9lTzwmgqHLGgA
!tar -xvf /home/test0/dataset/nasbench/NATS/NATS-sss-v1_0-50262-simple.tar -C /home/test0/dataset/nasbench/NATS/
```

If the file download fails, it is recommended to manually download it to Google Drive and then transfer it to Colab.
```bash
#If above part cant work,try another way to upload files, eg
from google.colab import drive
drive.mount('/content/drive')

# Original path
source_nasbench201 = "/content/drive/MyDrive/NAS-Bench-201-v1_1-096897.pth"
source_nats = "/content/drive/MyDrive/NATS-sss-v1_0-50262-simple.tar"

# Target path
target_dir = "/home/test0/dataset/nasbench/"
nats_dir = "/home/test0/dataset/nasbench/NATS/"



# Copy
!cp "{source_nasbench201}" {target_dir}
!cp "{source_nats}" {nats_dir}
!tar -xvf /home/test0/dataset/nasbench/NATS/NATS-sss-v1_0-50262-simple.tar -C /home/test0/dataset/nasbench/NATS/
```
---

### **Step P4: Clone the Survey Repository**

```bash
%cd /home/test0/
!git clone https://github.com/SLDGroup/survey-zero-shot-nas /home/test0/survey-zero-shot-nas
%cd /home/test0/survey-zero-shot-nas
```

---

### **Step P5: Install Dependencies**

```bash
!pip3 install nats_bench
# Clone NAS-Bench-201
!git clone https://github.com/D-X-Y/NAS-Bench-201.git
%cd NAS-Bench-201
# Install NAS-Bench-201 API
!pip3 install -e .
# Clone xautodl repository
!git clone https://github.com/D-X-Y/AutoDL-Projects.git
# Enter xautodl directory
%cd AutoDL-Projects
# Install xautodl package
!pip3 install -e .
!pip3 install ptflops
```

---

### **Step P6: Update Protobuf**

```bash
!apt-get remove -y protobuf-compiler
!wget https://github.com/protocolbuffers/protobuf/releases/download/v21.5/protoc-21.5-linux-x86_64.zip
!unzip protoc-21.5-linux-x86_64.zip -d protoc21
!mv protoc21/bin/protoc /usr/local/bin/
!chmod +x /usr/local/bin/protoc
```

---

### **Step P7: Check Protobuf Version**

```bash
!protoc --version
!find /home/test0 -name "*.proto"
```

---

### **Step P8: Compile Protobuf Files**

```bash
!protoc --proto_path=/home/test0/nasbench/nasbench/lib --python_out=/home/test0/nasbench/nasbench/lib /home/test0/nasbench/nasbench/lib/model_metrics.proto
!ls /home/test0/nasbench/nasbench/lib
```

---

### **Step P9: Install Specific Protobuf Version**

```bash
!pip install protobuf==3.20.3
# Restart after installation
```

---

### **Step P11: Check Python and Install Libraries**

```bash
!pip show protobuf

# Check Python version
!python --version

# Install required libraries
!pip install "torch>=1.2.0"
!pip install "numpy>=1.18.5"

# Create target directory and clone repository
%cd /home/test0
!git clone https://github.com/GATECH-EIC/HW-NAS-Bench.git

# Enter project directory
%cd /home/test0/HW-NAS-Bench
```

---

### **Step P12: Replace Files and Execute**

```bash
# Replace main and measure/__init__.py before execution
%cd /home/test0/survey-zero-shot-nas
!python main.py --searchspace=201 --dataset=cifar10 --data_path=/home/test0/dataset/ --metric=basic

%cd /home/test0/survey-zero-shot-nas
!python main.py --searchspace=201 --dataset=cifar10 --data_path=/home/test0/dataset/ --metric=lp

# 'basic' can be replaced with 'lp' or 'nrs'; however, 'nrs' requires an input file, usually 'basicGroup.txt' or 'lpGroup.txt'.

%cd /home/test0/survey-zero-shot-nas
!python main.py --searchspace=201 --dataset=cifar10 --data_path=/home/test0/dataset/ --metric=nrs --testName basicGroup.txt
!python main.py --searchspace=201 --dataset=cifar10 --data_path=/home/test0/dataset/ --metric=nrs --testName lpGroup.txt
```

---

