# CANet: A Commonness Aggregation Network with Cross-Modal for RGB-D Camouflaged Object Detection (CANet)
The paper is being submitted to The Visual Computer
1. Required libraries
python = 3.11.5 conda = 23.7.4 pytorch = 2.1.0 numpy = 1.24.3 torchvision = 0.16.0

2. Usage
Please follow the tips to download the datasets and pre-trained model:

Download RGB-D COD dataset from [https://pan.baidu.com/s/1MAh94WmBYl9HQaXG7orrVg?pwd=htrb 提取码：htrb]

Download RGB-D SOD dataset from [https://pan.baidu.com/s/12QntwlRaEc6ulcNXAhyVMg?pwd=ze4y 提取码：ze4y]

Download RGB-T SOD dataset from [https://pan.baidu.com/s/1Vw4ysRuVYP408t9zELXeqw?pwd=1219 提取码: 1219]

Download pretrained backbone weights from [https://pan.baidu.com/s/1gYoa542ET-DgdZA_kYcaMA?pwd=1219 提取码: 1219]

Training command :python train.py

Testing command :python test.py

Results: Qualitative results: we provide the prediction maps of RGB-D COD\RGB-D SOD\RGB-T SOD task, you can download them from [https://pan.baidu.com/s/1ppZI9KZY2XlPpC_jF7VtpQ?pwd=1219 提取码: 1219 ]

evaluation :python eval.py
