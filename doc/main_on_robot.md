```bash
pip install zmq
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers accelerate huggingface_hub pillow
export HF_ENDPOINT=https://hf-mirror.com
pip install rospkg catkin_pkg empy defusedxml PyYAML
```

```bash
roslaunch realsense2_camera rs_camera.launch align_depth:=true
```