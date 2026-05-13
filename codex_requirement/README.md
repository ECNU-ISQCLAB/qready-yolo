# Server Environment Setup

Run this after uploading and extracting the whole project on the server.

```bash
cd /path/to/yolov7-pytorch-master
python codex_requirement/setup_env.py
```

Default behavior:

- Creates conda env `yolov7-csdn`
- Installs Python 3.10
- Installs PyTorch + torchvision + CUDA 12.1 runtime
- Installs project Python dependencies
- Verifies CUDA, project files, weight compatibility, and one YOLO forward pass

Useful options:

```bash
python codex_requirement/setup_env.py --env-name yolov7-csdn
python codex_requirement/setup_env.py --cuda 11.8
python codex_requirement/setup_env.py --force
python codex_requirement/setup_env.py --skip-verify
```

For an A40 server, the default `--cuda 12.1` is usually appropriate if the NVIDIA driver is new enough. If CUDA verification fails because the driver is old, rerun with:

```bash
python codex_requirement/setup_env.py --force --cuda 11.8
```

After setup:

```bash
conda activate yolov7-csdn
python test_by_codex_5112056.py
python train.py
```
