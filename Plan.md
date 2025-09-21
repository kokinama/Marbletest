## Project Overview – “Heat‑Map‑Guided Grasp Detection + Tactile‑Slip Feedback on a Xilinx Kria KV260”

**Goal**  
Build a **complete, real‑time manipulation pipeline** that runs on a single embedded AI board (the Xilinx Kria KV260). The system must  

1. **Detect objects and predict a 6‑DoF grasp** using a heat‑map‑guided deep network (trained on the large GraspNet‑1B dataset).  
2. **Fuse tactile information** (8 × 8 pressure‑matrix) with the visual input by masking the image corners and inserting calibrated pressure values (Mask‑and‑Replace).  
3. **Adapt the camera view** when the initial centre patch does not yield a confident grasp – the “spring‑eye” creates four additional overlapping patches (left, right, up, down) and selects the best one.  
4. **Detect slip** with an ultra‑light CNN (≤ 1 ms on the Cortex‑R5) and **adjust the grasping force** in a gain‑scheduled PID loop (≤ 5 ms reaction).  
5. **Measure latency and power** (target ≤ 48 ms end‑to‑end, < 5 W total) and publish all code, data, and results as open‑source material.

The three students (CS‑A, CS‑B, DACS) will work **in parallel** on three largely independent work‑streams that intersect only at well‑defined integration points. Below is a **step‑by‑step description** of everything that will be built, why each component is needed, and how the work will be carried out.

---

## 1.  System Architecture (High‑Level Block Diagram)

```
+-------------------+      +-------------------+      +-------------------+
|   MIPI‑CSI Camera | ---> |   Pre‑Processing  | ---> |   Heat‑Map‑Guided |
|   (1280×960 RGB)  |      | (Crop, Blur,      |      |   Grasp Detector  |
+-------------------+      |  Mask‑and‑Replace)|      |   (ResNet‑34)     |
          |                |  + Unscharfe     |      +-------------------+
          |                |    Edge‑Patches) |                |
          v                v                  v                v
+-------------------+ +-------------------+ +-------------------+ +-------------------+
| 8×8 Pressure      | |   Spring‑Eye      | |   Slip‑CNN (R5)   | |   PID‑Force       |
| Matrix (I²C)      | | (5 Patches +     | | (2 DW‑Conv + FC) | |   Controller      |
+-------------------+ |  Scoring)        | +-------------------+ +-------------------+
          |                |                  |                |
          +----------------+------------------+----------------+
                           |
                           v
                +---------------------------+
                |   Motor / Servo Driver    |
                +---------------------------+
```

*All data moves through a **DMA‑based ring buffer** in the KV260 DDR‑4 memory, so the CPU never copies large buffers. The DPU (Deep Processing Unit) executes the visual model; the Cortex‑R5 runs the slip detector and the PID controller.*

---

## 2.  Hardware Platform

| Component | Model / Specs | Role in the System | Procurement |
|-----------|---------------|--------------------|-------------|
| **Kria KV260 Evaluation Board** | Xilinx Zynq‑MPSoC (Quad‑core Cortex‑A53 @1.5 GHz, Dual‑core Cortex‑R5, 4 GB DDR4) | Host CPU, DPU accelerator, real‑time core | Order 2 units (one development, one spare) during **M0** |
| **MIPI‑CSI Camera** | 1280 × 960 RGB, 30 fps (e.g., **Arducam IMX219**) | Provides high‑resolution visual input | Purchase together with KV260 board |
| **8 × 8 Pressure Matrix** | Piezo‑based, I²C, 200 Hz (e.g., **Interlink Force‑Touch 8×8**) | Supplies tactile pressure map for each grasp | Order during **M0** |
| **Power Monitor** | INA226 breakout board (0.1 mW resolution) | Measures instantaneous board power for profiling | Purchase early (used in **M14**) |
| **External NAS** | 12 TB RAID‑5 (USB‑3.0) | Stores the ~10 TB GraspNet‑1B dataset and intermediate files | Order before **M1** |
| **GPU Workstation** | RTX 4090 (or cloud GPU) | Trains the visual model and Slip‑CNN | Use university GPU cluster or cloud credits |

All hardware is installed in the **Maastricht campus lab** (available during teaching blocks; unavailable on holidays).

---

## 3.  Software Stack

| Layer | Library / Tool | Why it is used | Owner |
|-------|----------------|----------------|-------|
| **Operating System** | PetaLinux 2023.2 (Yocto‑based) | Provides real‑time kernel, device‑tree for KV260 peripherals | CS‑B |
| **Deep‑Learning Framework** | TensorFlow 2.12 (with `tf.keras`) | Training the heat‑map detector and Slip‑CNN | CS‑A (vision) & DACS (slip) |
| **Quantisation‑Aware Training** | TensorFlow‑Lite‑Micro + Vitis‑AI Quantizer | Generates INT8 models that run on the DPU and R5 without accuracy loss | CS‑A (vision) & DACS (slip) |
| **Vitis‑AI** | Vitis‑AI 2023.2 (DPU compiler, runtime) | Compiles the `.xmodel` for the KV260 DPU, provides C++ inference API | CS‑B |
| **Embedded NN Runtime** | CMSIS‑NN (for Cortex‑R5) | Executes the Slip‑CNN in < 1 ms on the real‑time core | DACS |
| **Computer Vision** | OpenCV 4.8 (with NEON intrinsics) | Fast image cropping, Gaussian blur, Mask‑and‑Replace implementation | CS‑B |
| **Physics Simulation** | PyBullet 3.2 (Python) | Generates synthetic slip data (different friction coefficients, object masses) | DACS |
| **Version Control & CI** | GitHub + GitHub Actions (Docker) | Guarantees reproducibility, automatic build of Docker images for preprocessing and training | CS‑B (repo setup) |
| **Documentation & DOI** | Zenodo (for dataset) + ReadTheDocs (for code docs) | Open‑source release with permanent DOI | DACS (dataset) & CS‑B (code) |
| **Plotting / Statistics** | Python `matplotlib`, `seaborn`, `scipy.stats` | Generates figures for the paper and performs paired‑t tests, Cohen’s d | CS‑A (visual) & DACS (tactile) |

All code will be stored in a **single GitHub repository** with three top‑level directories:

```
/data_preprocess   # GraspNet‑1B handling, synthetic tactile generation
/vision_model      # Heat‑map network, training scripts, QAT, export
/embedded_firmware # KV260 firmware (C/C++), Mask‑and‑Replace, Spring‑Eye, PID
/tactile_slip      # Pressure‑matrix driver, Slip‑CNN, calibration scripts
/docs              # LaTeX paper, README, hardware schematics
```

---

## 4.  Detailed Work‑Stream Breakdown  

Below each work‑stream is described step‑by‑step, with **why** the step is needed, **what** will be produced, and **where** to find reference material.

### 4.1.  CS‑A – Vision & Machine‑Learning  

| Step | Action | Output | Reference / Link |
|------|--------|--------|-------------------|
| **4.1.1** | **Download GraspNet‑1B** (≈ 10 TB) from the official site. Use `rsync` or `wget` with checksum verification. | Raw RGB‑Depth‑Pose files on NAS. | https://graspnet.net/ |
| **4.1.2** | **Down‑sample** all RGB images to **640 × 480** (the KV260 DPU input size). Store a CSV with image‑id → file‑path mapping. | Smaller image set (≈ 2 TB) + index file. | OpenCV `cv2.resize` (NEON‑accelerated). |
| **4.1.3** | **Generate synthetic tactile samples**: for each image, create a random 8 × 8 pressure map (Gaussian‑blurred, scaled to realistic force range). Store as `.npy`. | `tactile_{id}.npy` files. | Simple NumPy script (`np.random.randn`). |
| **4.1.4** | **Convert GraspNet pose annotations** to **heat‑map format** (as in Dex‑Net 2.0). For each grasp, draw a 2‑D Gaussian centred at the pixel location, orientation encoded in a separate channel. | `heatmap_{id}.npz` (3‑channel: grasp‑center, angle‑sin, angle‑cos). | https://github.com/BerkeleyAutomation/dex-net |
| **4.1.5** | **Define the model**: ResNet‑34 backbone (pre‑trained on ImageNet) + a 1 × 1 convolution head that outputs the 3‑channel heat‑map. Use `tf.keras.applications.ResNet34`. | `model.py` (TensorFlow). | https://keras.io/api/applications/resnet/ |
| **4.1.6** | **Train the model** on the down‑sampled GraspNet data (batch = 32, Adam lr = 1e‑4, 200 epochs). Use **mixed‑precision** (`tf.keras.mixed_precision`). | Trained checkpoint (`ckpt/`). | TensorFlow mixed‑precision guide. |
| **4.1.7** | **Quantisation‑Aware Training (QAT)**: insert fake‑quant layers (`tf.quantization.fake_quant_with_min_max_vars`) and continue training for 10 epochs. | QAT‑ready checkpoint. | https://www.tensorflow.org/model_optimization/guide/quantization/training |
| **4.1.8** | **Export to Vitis‑AI**: convert the QAT checkpoint to a TensorFlow‑Lite model (`tflite_convert --output_format=VAI`), then run the Vitis‑AI compiler (`vai_c_xir`). | `heatmap_grasp.xmodel`. | https://github.com/Xilinx/Vitis-AI |
| **4.1.9** | **Validate INT8 accuracy** on a held‑out set (target mAP ≥ 0.78). If loss > 2 % re‑train QAT with per‑channel scaling. | Accuracy report (`accuracy.txt`). | Vitis‑AI Model Zoo evaluation script. |
| **4.1.10** | **Write visual‑only ablation scripts** (baseline centre‑patch only). | `run_baseline.py`. | – |
| **4.1.11** | **Prepare figures & tables** for the paper (training curves, heat‑map examples). | `figures/`. | Matplotlib. |

### 4.2.  CS‑B – Embedded Software & System Integration  

| Step | Action | Output | Reference / Link |
|------|--------|--------|-------------------|
| **4.2.1** | **Set up PetaLinux** on the KV260 (create a BSP, enable I²C, MIPI‑CSI, DMA). | Bootable image (`image.ub`). | https://xilinx.github.io/kria-apps-docs/ |
| **4.2.2** | **Write a driver** for the 8 × 8 pressure matrix (I²C read at 200 Hz, store in a circular buffer). | `pressure_driver.c`. | Linux I²C documentation. |
| **4.2.3** | **Implement Mask‑and‑Replace**: after each frame is captured, mask the four 80 × 80 corners, up‑sample the 8 × 8 pressure map (bilinear) and copy it into the masked region. Use NEON intrinsics for the up‑sampling (`vld1q_f32`, `vmlaq_f32`). | `mask_replace.c`. | ARM NEON Intrinsics Guide. |
| **4.2.4** | **Add Unscharfe Edge‑Patches**: take the full‑resolution image, apply a Gaussian blur (`cv::GaussianBlur`), down‑sample to 640 × 480, up‑sample back, and paste into the same corners. Provide a compile‑time flag to enable/disable the blur. | `blur_edge.c`. | OpenCV `GaussianBlur`. |
| **4.2.5** | **Create the Spring‑Eye logic**: generate five 640 × 480 patches (centre, left, right, up, down) by adjusting the crop offsets (± 80 px). For each patch, call the DPU inference API (`dpuRunTask`) and collect the **Grasp‑Score** (see Section 5). | `spring_eye.c`. | Vitis‑AI Runtime API. |
| **4.2.6** | **Design the Grasp‑Score function**: <br> `score = w1 * mAP + w2 * (1 / pose_error) + w3 * safety_margin - w4 * edge_penalty`. The weights (`w1…w4`) are tuned on a small validation set (grid‑search). | `score.c`. | – |
| **4.2.7** | **Implement the DMA Ring‑Buffer**: allocate two buffers in DDR (one for raw RGB frames, one for processed frames). Use the Xilinx DMA driver (`xdma`) to move data from the CSI capture engine to DDR without CPU copies. | `ring_buffer.c`. | Xilinx DMA driver documentation. |
| **4.2.8** | **Integrate the DPU Job‑Queue**: enqueue the five patches sequentially; while the DPU processes patch i, the CPU prepares patch i+1 (crop, mask, blur). This overlapping reduces total inference time to ≈ 30 ms. | `dpu_queue.c`. | Vitis‑AI Multi‑Tasking Example. |
| **4.2.9** | **Port the Slip‑CNN** (see DACS) to the Cortex‑R5 using CMSIS‑NN. Create an ISR that runs at 1 kHz, reads the latest pressure map, runs the Slip‑CNN, and sets a global `slip_flag`. | `slip_r5.c`. | CMSIS‑NN User Guide. |
| **4.2.10** | **Implement Gain‑Scheduled PID**: when `slip_flag == 1`, increase `Kp` by 30 % and `Ki` by 10 %; otherwise use nominal gains. The controller outputs a PWM duty cycle for the servo driver. | `pid_controller.c`. | Classic PID design (e.g., Åström & Murray). |
| **4.2.11** | **Full‑pipeline main loop** (pseudo‑code): <br>```c <br>while(1){ <br>  capture_frame(); <br>  preprocess_mask_replace(); <br>  run_spring_eye(); <br>  run_slip_cnn(); <br>  update_pid(); <br>  command_motor(); <br>} ``` | `main.c`. | – |
| **4.2.12** | **Create Docker image** that builds the firmware, flashes the KV260, and runs a test harness (captures a frame, prints latency). | `Dockerfile` (in repo). | Docker documentation. |
| **4.2.13** | **Write integration tests** (GoogleTest) that verify: <br>• Correct mask placement <br>• Proper patch offsets <br>• PID output range | `tests/`. | GoogleTest tutorial. |

### 4.3.  DACS – Tactile Sensing, Slip Detection & Evaluation  

| Step | Action | Output | Reference / Link |
|------|--------|--------|-------------------|
| **4.3.1** | **Assemble the pressure matrix** on a breakout board, connect to KV260 I²C pins, and verify communication with `i2cget`. | Working hardware, `i2c` address. | Linux I²C tools. |
| **4.3.2** | **Calibrate the matrix**: place known weights (e.g., 10 g, 50 g, 200 g) on each cell, record raw ADC values, fit a linear model per cell (`force = a·raw + b`). Store calibration coefficients in a JSON file. | `calibration.json`. | Least‑squares fitting (`numpy.linalg.lstsq`). |
| **4.3.3** | **Create a data‑logger** that records synchronized RGB frames and calibrated pressure maps at 30 Hz (timestamped with `clock_gettime(CLOCK_MONOTONIC)`). Save to a binary log (`.bag` or custom format). | `dataset/recordings/*.log`. | ROS2 bag format (optional). |
| **4.3.4** | **Generate synthetic slip data** using PyBullet: <br>• Load a simple gripper model and a set of objects (cylinder, sphere, box). <br>• Randomly vary friction coefficients (0.2 – 0.9) and apply a closing force. <br>• Record the pressure map at each simulation step and label frames as *slip* (if relative motion > 1 mm) or *stable*. | `synthetic_slip/*.npz`. | https://github.com/bulletphysics/bullet3 |
| **4.3.5** | **Design the Slip‑CNN**: two depth‑wise 3 × 3 convolutions (ReLU + BatchNorm) followed by a fully‑connected layer with a sigmoid output. Total parameters ≈ 1 k. | `slip_model.py`. | TensorFlow‑Lite‑Micro example. |
| **4.3.6** | **Train Slip‑CNN** on the synthetic slip dataset (batch = 64, Adam lr = 1e‑3, 30 epochs). Validate on a small set of **real** slip recordings (collected with the hardware). Aim for **F1 ≥ 0.90**. | Trained checkpoint (`slip_ckpt/`). | – |
| **4.3.7** | **Quantisation‑Aware Training** for Slip‑CNN (same procedure as CS‑A). Export to TensorFlow‑Lite‑Micro (`.tflite`) and then to CMSIS‑NN (`.c` source). | `slip_int8.c`. | Vitis‑AI Quantizer for microcontrollers. |
| **4.3.8** | **Integrate Slip‑CNN on the Cortex‑R5** (see CS‑B step 4.2.9). Verify inference time < 1 ms using the `cycle_counter`. | Pass/fail report (`slip_latency.txt`). | CMSIS‑NN benchmark. |
| **4.3.9** | **Power & Latency Profiling**: use the INA226 to log voltage/current at 1 kHz while running the full pipeline. Export CSV for analysis. | `power_log.csv`. | INA226 datasheet. |
| **4.3.10** | **Run Ablation Studies**: <br>• Baseline (vision only) <br>• +Tactile (Mask‑and‑Replace) <br>• +Spring‑Eye <br>• +Unscharfe Edge‑Patches <br>Collect **Grasp‑Success‑Rate**, **Pose‑Error**, **Peak‑Force**, **Latency**, **Power** for each configuration. | `ablation_results.xlsx`. | – |
| **4.3.11** | **Statistical analysis**: paired‑t test between each configuration, compute Cohen’s d. Write a short report (`stats_report.pdf`). | `stats_report.pdf`. | SciPy `ttest_rel`. |
| **4.3.12** | **Prepare dataset for release**: include a subset of synchronized RGB‑pressure pairs (≈ 5 k samples), calibration JSON, and a README describing the format. Upload to Zenodo, obtain DOI. | Zenodo DOI (e.g., `10.5281/zenodo.XXXXXX`). | https://zenodo.org/ |

---

## 5.  Grasp‑Score Computation (used by Spring‑Eye)

The **Grasp‑Score** decides which of the five patches (centre + four neighbours) is the best candidate.

```
score = w1 * visual_confidence
        + w2 * (1 / pose_error)          # lower reprojection error → higher score
        + w3 * safety_margin             # distance to known obstacles (from blurred edge patches)
        - w4 * edge_penalty              # penalise grasps that lie too close to the image border
```

* **visual_confidence** = maximum heat‑map value (after soft‑max).  
* **pose_error** = Re‑Projection Error (RPE) computed from the 6‑DoF pose returned by the heat‑map network.  
* **safety_margin** = minimum distance (in pixels) from the grasp centre to any high‑intensity region in the blurred edge patch (acts as a proxy for obstacles).  
* **edge_penalty** = 1 if the grasp centre lies within 20 px of the image border, else 0.

The weights (`w1…w4`) are tuned on a small validation set (grid‑search, see CS‑A step 4.1.9). The final values are stored in a header file (`score_weights.h`) and can be changed without recompiling the firmware.

---

## 6.  Evaluation & Expected Results  

| Metric | Target (after full system) | How it is measured |
|--------|---------------------------|--------------------|
| **Grasp‑Success‑Rate** (objects placed correctly) | **≥ 90 %** overall; **≥ 92 %** for objects that start at the image border | Automated test rig with 3 object classes (soft tomato, rigid glass bottle, slippery metal flask). |
| **Pose‑Error** (average RPE) | **≤ 3.5 mm** | Computed from the 6‑DoF pose output vs. ground‑truth pose (from motion‑capture). |
| **Peak‑Contact‑Force** (during grasp) | **≤ 30 %** of the nominal force for soft objects (thanks to slip‑based PID) | Force sensor on the gripper (calibrated). |
| **End‑to‑End Latency** (capture → motor command) | **≤ 48 ms** (≈ 20 ms for vision, ≤ 5 ms for slip detection, ≤ 5 ms for PID, rest for DMA & CPU) | Timestamp each stage with `clock_gettime` and log to CSV. |
| **Power Consumption** (average during a grasp) | **< 5 W** (≈ 3 W for DPU inference, 0.8 W for R5, 0.5 W for peripherals) | INA226 readings, averaged over 10 s runs. |
| **Slip‑Detection F1‑Score** | **≥ 0.90** | Confusion matrix on real slip recordings. |

All numbers will be reported with **95 % confidence intervals** and compared against a **baseline** (vision‑only, no tactile, no spring‑eye).

---

## 7.  Open‑Source Release  

* **GitHub repository** – public under the **BSD‑3‑Clause** license.  
  * URL (to be created): `https://github.com/your‑group/kv260‑grasp‑tactile`  
* **Docker images** – hosted on Docker Hub (`your‑docker‑id/kv260‑grasp`).  
* **Dataset** – a curated subset of GraspNet‑1B + synchronized tactile maps, uploaded to **Zenodo** with a permanent DOI.  
* **Documentation** – generated with **ReadTheDocs**, includes hardware schematics, build instructions, API reference, and a quick‑start guide.  

These resources will allow any researcher with a KV260 (or a similar MPSoC) to reproduce the results and extend the system.

---

## 8.  Key Reference Links  

| Topic | Link |
|-------|------|
| **GraspNet‑1B dataset** | https://graspnet.net/ |
| **Dex‑Net 2.0 (heat‑map grasp representation)** | https://github.com/BerkeleyAutomation/dex-net |
| **ResNet‑34 (Keras implementation)** | https://keras.io/api/applications/resnet/ |
| **TensorFlow Model Optimization (QAT)** | https://www.tensorflow.org/model_optimization/guide/quantization/training |
| **Vitis‑AI (DPU compiler & runtime)** | https://github.com/Xilinx/Vitis-AI |
| **CMSIS‑NN (micro‑controller inference)** | https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/NN |
| **OpenCV with NEON intrinsics** | https://developer.arm.com/architectures/instruction-sets/simd-isas/neon |
| **PyBullet (physics simulation)** | https://github.com/bulletphysics/bullet3 |
| **INA226 Power Monitor** | https://www.ti.com/product/INA226 |
| **Docker Hub** | https://hub.docker.com/ |
| **Zenodo (data DOI)** | https://zenodo.org/ |
| **ReadTheDocs** | https://readthedocs.org/ |
| **GoogleTest (C++ unit testing)** | https://github.com/google/googletest |
| **Mixed‑Precision Training guide** | https://www.tensorflow.org/guide/mixed_precision |
| **PID design (Åström & Murray)** | https://www.springer.com/gp/book/9781441916650 |

---

## 9.  Summary of What Will Be Built  

| Component | What it does | Where the code lives |
|-----------|--------------|----------------------|
| **Heat‑Map‑Guided Grasp Detector** | Takes a 640 × 480 RGB image, outputs a 3‑channel heat‑map that encodes grasp centre and orientation. | `vision_model/` |
| **Mask‑and‑Replace + Unscharfe Edge‑Patches** | Replaces the four image corners with calibrated pressure values; optionally adds a low‑frequency blurred context. | `embedded_firmware/mask_replace.c` |
| **Spring‑Eye Multi‑Region Scanning** | Generates five overlapping patches, runs the visual model on each, computes a Grasp‑Score, selects the best patch. | `embedded_firmware/spring_eye.c` |
| **Slip‑CNN (R5)** | Classifies a 8 × 8 pressure map as *slipping* or *stable* in < 1 ms. | `tactile_slip/slip_r5.c` |
| **Gain‑Scheduled PID Controller** | Adjusts the motor PWM based on the slip flag to increase force only when needed. | `embedded_firmware/pid_controller.c` |
| **DMA Ring‑Buffer & DPU Job‑Queue** | Moves frames from the camera to DDR without CPU copies; queues up to five DPU inference jobs to overlap compute and memory transfers. | `embedded_firmware/ring_buffer.c` & `dpu_queue.c` |
| **Power & Latency Measurement Suite** | Logs voltage/current (INA226) and timestamps for each pipeline stage; produces CSV files for analysis. | `tactile_slip/power_latency.c` |
| **Dataset & Calibration Package** | Synchronized RGB‑pressure pairs, calibration JSON, synthetic slip data, and documentation. | `data_preprocess/` (uploaded to Zenodo) |
| **Paper & Presentation Materials** | LaTeX source, figures, tables, demo video. | `docs/` |

All of these pieces will be **integrated** into a single firmware binary that can be flashed onto the KV260 and run **autonomously**: the robot arm sees an object, decides where to grasp (even if the object is at the image edge), monitors tactile slip, and adapts its force in real time—all while staying under 5 W power budget and below 50 ms latency.

---

### Next Steps for the Team  

1. **M0 (Week 38)** – Order hardware, set up the GitHub repo, create the initial PetaLinux BSP.  
2. **M1 & M2 (Weeks 41‑42)** – CS‑A starts GraspNet download & preprocessing; DACS assembles and tests the pressure matrix.  
3. **M3‑M5 (Weeks 43‑48)** – CS‑A trains and quantises the heat‑map model; DACS finishes calibration.  
4. **M6‑M9 (Weeks 5‑7 of 2026)** – CS‑B implements Mask‑and‑Replace, Spring‑Eye, and the DMA pipeline; DACS generates synthetic slip data and trains the Slip‑CNN.  
5. **M10‑M14 (Weeks 10‑15)** – Full integration, profiling, and ablation studies.  
6. **M15‑M19 (Weeks 18‑24)** – Write the paper, submit to ICRA/IROS, record the demo video, and publish the open‑source release.

With this detailed roadmap, each student knows **exactly what to build, why it matters, and how to do it**. The project is now ready to move forward, respecting the academic calendar, keeping the workload balanced, and delivering a publishable, reproducible research contribution.