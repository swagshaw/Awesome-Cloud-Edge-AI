[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![GitHub](https://img.shields.io/badge/License-MIT-lightgrey.svg)

# Awesome-System-for-Could-and-Edge-AI
A curated list of research in System for Edge Intelligence and Computing(Edge MLSys), including Frameworks, Tools, Repository, etc. Paper notes are also provided.

## Contents
- [Tutorial and video](#tutorial-and-video)
- [Project](#project)
- [Survey](#survey)
- [Blog](#blog)
- [Cloud-Edge Collaborative Training](#cloud-edge-collaborative-training)
- [Edge-Caching for Sharing of DL Computation](#edge-caching-for-sharing-of-dl-computation)
- [Cloud-Edge Collaborative Inference](#cloud-edge-collaborative-inference)
- [Selection and Optimization of DL Models in Edge](#selection-and-optimization-of-dl-models-in-edge)
- [Real-time Applications Based on Edge-Cloud Intelligence](#real-time-applications-based-on-edge-cloud-intelligence)
- [Fog AI ](#fog-ai)

## General Resources
### Tutorial and Video
- Overview of edge computing in LinkedIn. [[LinkedIn]](https://www.linkedin.com/learning/iot-foundations-operating-system-applications/overview-of-edge-computing?autoAdvance=true&autoSkip=false&autoplay=true&resume=true&u=43752620)
- IoT (Internet of Things) Wireless & Cloud Computing Emerging Technologies. [[Coursera]](https://www.coursera.org/lecture/iot-wireless-cloud-computing/5-10-edge-computing-pOK8T)
- Udemy Introduction to Edge Computing. [[Udemy]](https://www.udemy.com/course/introduction-to-edge-computing/)
- IOT Edge Computing | IoT Examples | Use Cases | HackerEarth Webinar.[[Youtube]](https://www.youtube.com/watch?v=Xm8frqTZRVI)
- Intel® Edge AI for IoT Developers [[Udacity]](https://www.udacity.com/course/intel-edge-ai-for-iot-developers-nanodegree--nd131)
- Stanford Seminar - The Future of Edge Computing from an International Perspective. [[Youtube]](https://www.youtube.com/watch?v=Hhobq4fs87w)
### Project:

- deepC is a vendor independent deep learning library, compiler and inference framework designed for small form-factor devices including μControllers, IoT and Edge devices[[GitHub]](https://github.com/ai-techsystems/deepC)
- Tengine, developed by OPEN AI LAB, is an AI application development platform for AIoT scenarios launched by OPEN AI LAB, which is dedicated to solving the fragmentation problem of aiot industrial chain and accelerating the landing of AI industrialization. [[GitHub]](https://github.com/OAID/Tengine)
- Mobile Computer Vision @ Facebook [[GitHub]](https://github.com/facebookresearch/mobile-vision)
- alibaba/MNN: MNN is a lightweight deep neural network inference engine. It loads models and do inference on devices. [[GitHub]](https://github.com/alibaba/MNN)
- XiaoMi/mobile-ai-bench: Benchmarking Neural Network Inference on Mobile Devices [[GitHub]](https://github.com/XiaoMi/mobile-ai-bench)
- XiaoMi/mace-models: Mobile AI Compute Engine Model Zoo [[GitHub]](https://github.com/XiaoMi/mace-models)
- Tencent/nccn: ncnn is a high-performance neural network inference computing framework optimized for mobile platforms. [[Github]](https://github.com/Tencent/ncnn)
- Tencent/TNN: [[Github]](https://github.com/Tencent/TNN)
- SqueezeWave: Extremely Lightweight Vocoders for On-device Speech Synthesis [[GitHub]](https://github.com/tianrengao/SqueezeWave)[[Paper]](https://arxiv.org/abs/2001.05685)
- Kubeedge: A Kubernetes Native Edge Computing Framework [[GitHub]](https://github.com/kubeedge/kubeedge)
### Survey:

-  Convergence of edge computing and deep learning: A comprehensive survey. [[Paper]](https://arxiv.org/pdf/1907.08349)
    - Wang, X., Han, Y., Leung, V. C., Niyato, D., Yan, X., & Chen, X. (2020).
    - IEEE Communications Surveys & Tutorials, 22(2), 869-904.
- Deep learning with edge computing: A review. [[Paper]](https://www.cs.ucr.edu/~jiasi/pub/deep_edge_review.pdf)
    - Chen, J., & Ran, X. 
    - Proceedings of the IEEE, 107(8), 1655-1674.(2019). 
- Edge Intelligence: Paving the Last Mile of Artificial Intelligence with Edge Computing. [[Paper]](https://arxiv.org/pdf/1905.10083.pdf)
    - Zhou, Z., Chen, X., Li, E., Zeng, L., Luo, K., & Zhang, J.
    - arXiv: Distributed, Parallel, and Cluster Computing. (2019). 
- Machine Learning at Facebook: Understanding Inference at the Edge. [[Paper]](https://ieeexplore.ieee.org/abstract/document/8675201)
    - Wu, C., Brooks, D., Chen, K., Chen, D., Choudhury, S., Dukhan, M., ... & Zhang, P. 
    - high-performance computer architecture.(2019). 
- Mobile Edge Computing: A Survey on Architecture and Computation Offloading. [[Paper]](https://ieeexplore.ieee.org/document/7879258)
    - P. Mach and Z. Becvar.
    - IEEE Communications Surveys & Tutorials, vol. 19, no. 3, pp. 1628-1656, thirdquarter 2017


### Blog

- 边缘智能综述（edge intelligence)[[Zhihu]](https://zhuanlan.zhihu.com/p/145439319)

- AI edge engineer [[Blog]](https://docs.microsoft.com/en-us/learn/paths/ai-edge-engineer/)

- Advance your edge computing skills with three new AWS Snowcone courses[[Blog]](https://aws.amazon.com/cn/blogs/training-and-certification/advance-your-edge-computing-skills-with-three-new-aws-snowcone-courses/)
- How fast is my model? [[Blog]](https://machinethink.net/blog/how-fast-is-my-model/)
## Paper:

### Cloud-Edge Collaborative Training

- Collaborative learning between cloud and end devices: an empirical study on location prediction. [[Paper]](https://www.microsoft.com/en-us/research/uploads/prod/2019/08/sec19colla.pdf)
    - Lu, Y., Shu, Y., Tan, X., Liu, Y., Zhou, M., Chen, Q., & Pei, D.
    - ACM/IEEE Symposium on Edge Computing(2019)
- Distributed Machine Learning through Heterogeneous Edge Systems.[[Paper]](https://i2.cs.hku.hk/~cwu/papers/hphu-aaai20.pdf)
    - Hu, H., Wang, D., & Wu, C. (2020). 
    - In AAAI (pp. 7179-7186).

### Cloud-Edge Collaborative Inference
- Modeling of Deep Neural Network (DNN) Placement and Inference in Edge Computing. [[Paper]](https://arxiv.org/pdf/2001.06901.pdf)
    - Bensalem, M., Dizdarević, J. and Jukan, A., 2020.
    - arXiv preprint arXiv:2001.06901. 
- Characterizing the Deep Neural Networks Inference Performance of Mobile Applications. [[Paper]](https://arxiv.org/pdf/1909.04783.pdf)
    - Ogden, S.S. and Guo, T., 2019.
    - arXiv preprint arXiv:1909.04783.
- Neurosurgeon: Collaborative intelligence between the cloud and mobile edge. [[Paper]](http://web.eecs.umich.edu/~jahausw/publications/kang2017neurosurgeon.pdf)
    - Kang, Y., Hauswald, J., Gao, C., Rovinski, A., Mudge, T., Mars, J. and Tang, L., 2017, April. 
    - In ACM SIGARCH Computer Architecture News (Vol. 45, No. 1, pp. 615-629). ACM.
    - [[My note]](https://c7i8iaoaz8.larksuite.com/docs/docusjAzPzX9S0MyRa3iaex6Ewd)
- 26ms Inference Time for ResNet-50: Towards Real-Time Execution of all DNNs on Smartphone [[Paper]](https://arxiv.org/pdf/1905.00571.pdf)
    - Wei Niu, Xiaolong Ma, Yanzhi Wang, Bin Ren (*ICML2019*)
- Big/little deep neural network for ultra low power inference.[[Paper]](https://ieeexplore.ieee.org/document/7331375?reload=true)
    - Park, E., Kim, D. Y., Kim, S., Kim, Y. M., Kim, G., Yoon, S., & Yoo, S.
    - international conference on hardware/software codesign and system synthesis.(2015)
- JointDNN: an efficient training and inference engine for intelligent mobile cloud computing services. [[Paper]](https://arxiv.org/pdf/1801.08618.pdf)
    - Eshratifar, A. E., Abrishami, M. S., & Pedram, M.  
    - IEEE Transactions on Mobile Computing.(2019).
- TeamNet: A Collaborative Inference Framework on the Edge. 
    - Fang, Y., Jin, Z., & Zheng, R.
    - In 2019 IEEE 39th International Conference on Distributed Computing Systems (ICDCS) (pp. 1487-1496). IEEE. (2019, July). 
- Bottlenet++: An end-to-end approach for feature compression in device-edge co-inference systems. [[Paper]](https://arxiv.org/pdf/1910.14315.pdf)
    - Shao, J., & Zhang, J. 
    - In 2020 IEEE International Conference on Communications Workshops (ICC Workshops) (pp. 1-6). IEEE.(2020, June). 
- Distributing deep neural networks with containerized partitions at the edge. [[Paper]](https://www.usenix.org/system/files/hotedge19-paper-zhou.pdf)
    - Zhou, L., Wen, H., Teodorescu, R., & Du, D. H. (2019). 
    - In 2nd {USENIX} Workshop on Hot Topics in Edge Computing (HotEdge 19).
 - Dynamic adaptive DNN surgery for inference acceleration on the edge. [[Paper]](https://ieeexplore.ieee.org/abstract/document/8737614/)
    - Hu, C., Bao, W., Wang, D., & Liu, F. (2019, April). 
    - In IEEE INFOCOM 2019-IEEE Conference on Computer Communications (pp. 1423-1431). IEEE.
- Collaborative execution of deep neural networks on internet of things devices. [[Paper]](https://arxiv.org/pdf/1901.02537)
    - Hadidi, R., Cao, J., Ryoo, M. S., & Kim, H. 
    - arXiv preprint arXiv:1901.02537.(2019). 
- DeepThings: Distributed Adaptive Deep Learning Inference on Resource-Constrained IoT Edge Clusters. [[Paper]](https://ieeexplore.ieee.org/document/8493499)
    - Zhao, Z., Barijough, K. M., & Gerstlauer, A. 
    - IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 37(11), 2348-2359.(2018). 
- Scaling for edge inference of deep neural networks. [[Paper]](https://www.nature.com/articles/s41928-018-0059-3)
    - Bingqian Lu, Jianyi Yang, Shaolei Ren
    - ACM/IEEE Symposium on Edge Computing 2020
- Swing: Swarm Computing for Mobile Sensing.[[Paper]](http://people.duke.edu/~bcl15/documents/fan18-icdcs.pdf)
    - Fan, S., Salonidis, T., & Lee, B. C.  
    - international conference on distributed computing systems(2018).[[Paper]](
- A Locally Distributed Mobile Computing Framework for DNN based Android Applications.[[Paper]](https://dl.acm.org/doi/10.1145/3275219.3275236)
    - Jiajun Zhang, Shihong Chen, Bichun Liu, Yun Ma, Xing Chen
    - Internetware 2018: 17:1-17:6
- Auto-tuning Neural Network Quantization Framework for Collaborative Inference Between the Cloud and Edge.[[Paper]](https://arxiv.org/abs/1812.06426)
    - Guangli Li, Lei Liu, Xueying Wang, Xiao Dong, Peng Zhao, Xiaobing Feng
    - ICANN (1) 2018: 402-411
- DeepX: a software accelerator for low-power deep learning inference on mobile devices.[[Paper]](https://ieeexplore.ieee.org/document/7460664)
    - Nicholas D. Lane, Sourav Bhattacharya, Petko Georgiev, Claudio Forlivesi, Lei Jiao, Lorena Qendro, Fahim Kawsar
    - IPSN 2016: 23:1-23:12
- ECRT: An Edge Computing System for Real-Time Image-based Object Tracking. [[Paper]](https://dl.acm.org/doi/10.1145/3274783.3275199)
    - Zhihe Zhao, Zhehao Jiang, Neiwen Ling, Xian Shuai, Guoliang Xing
    - SenSys 2018: 394-395
- Learning IoT in Edge: Deep Learning for the Internet of Things with Edge Computing[[Paper]](https://ieeexplore.ieee.org/document/8270639)
    - He Li, Kaoru Ota, Mianxiong Dong
    - IEEE Netw. 32(1): 96-101 (2018)
### Edge-Caching for Sharing of DL Computation
- Hierarchical Edge Caching in Device-to-Device Aided Mobile Networks: Modeling, Optimization, and Design. [[Paper]](https://ieeexplore.ieee.org/document/8374077)
    - Xiuhua Li, Xiaofei Wang, Peng-Jun Wan, Zhu Han, Victor C. M. Leung
    -  IEEE J. Sel. Areas Commun. 36(8): 1768-1785 (2018)
- DeepCachNet: A Proactive Caching Framework Based on Deep Learning in Cellular Networks[[Paper]](https://ieeexplore.ieee.org/document/8642795)
    - Shailendra Rathore, Jung Hyun Ryu, Pradip Kumar Sharma, Jong Hyuk Park
    -  IEEE Netw. 33(3): 130-138 (2019)
- Deep learning-based edge caching for multi-cluster heterogeneous networks[[Paper]](Deep learning-based edge caching for multi-cluster heterogeneous networks)
    - 	Jiachen Yang, Jipeng Zhang, Chaofan Ma, Huihui Wang, Juping Zhang, Gan Zheng
    - 	Neural Comput. Appl. 32(19): 15317-15328 (2020)
- Learn to Cache: Machine Learning for Network Edge Caching in the Big Data Era[[Paper]](https://ieeexplore.ieee.org/document/8403948)
    - Zheng Chang, Lei Lei, Zhenyu Zhou, Shiwen Mao, Tapani Ristaniemi
    - IEEE Wirel. Commun. 25(3): 28-35 (2018)
- Deep Learning Based Caching for Self-Driving Car in Multi-access Edge Computing[[Paper]](https://arxiv.org/abs/1810.01548)
    - 	Anselme Ndikumana, Nguyen H. Tran, DoHyeon Kim, Ki Tae Kim, Choong Seon Hong
    - 	IEEE Trans. Intell. Transp. Syst. 22(5): 2862-2877 (2021)
- A smart caching mechanism for mobile multimedia in information centric networking with edge computing[[Paper]](https://www.sciencedirect.com/science/article/abs/pii/S0167739X18311841?via%3Dihub)
    - 	Yayuan Tang, Kehua Guo, Jianhua Ma, Yutong Shen, Tao Chi
    - 	Future Gener. Comput. Syst. 91: 590-600 (2019)
### Selection and Optimization of DL Models in Edge
- Context-Aware Convolutional Neural Network over Distributed System in Collaborative Computing. [[Paper]](https://dl.acm.org/doi/10.1145/3316781.3317792)
    - Choi, J., Hakimi, Z., Shin, P. W., Sampson, J., & Narayanan, V. (2019). 
    - design automation conference.
- OpenEI: An Open Framework for Edge Intelligence. [[Paper]](https://arxiv.org/pdf/1906.01864.pdf)
    - Zhang, X., Wang, Y., Lu, S., Liu, L., Xu, L., & Shi, W. 
    - international conference on distributed computing systems.(2019). 
### Real-time Applications Based on Edge-Cloud Intelligence
- Latency and Throughput Characterization of Convolutional Neural Networks for Mobile Computer Vision [[Paper]](https://arxiv.org/pdf/1803.09492.pdf)
    - Hanhirova, J., Kämäräinen, T., Seppälä, S., Siekkinen, M., Hirvisalo, V. and Ylä-Jääski
    - In Proceedings of the 9th ACM Multimedia Systems Conference (pp. 204-215).
- NestDNN: Resource-Aware Multi-Tenant On-Device Deep Learning for Continuous Mobile Vision [[Paper]](https://arxiv.org/pdf/1810.10090.pdf)
    - Fang, Biyi, Xiao Zeng, and Mi Zhang. (*MobiCom 2018*)
    - Summary: Borrow some ideas from network prune. The pruned model then recovers to trade-off computation resource and accuracy at runtime
- Lavea: Latency-aware video analytics on edge computing platform [[Paper]](http://www.cs.wayne.edu/~weisong/papers/yi17-LAVEA.pdf)
    - Yi, Shanhe, et al. (*Second ACM/IEEE Symposium on Edge Computing. ACM, 2017.*)
- Scaling Video Analytics on Constrained Edge Nodes [[Paper]](http://www.sysml.cc/doc/2019/197.pdf) [[GitHub]](https://github.com/viscloud/filterforward)
    - Canel, C., Kim, T., Zhou, G., Li, C., Lim, H., Andersen, D. G., Kaminsky, M., and Dulloo (*SysML 2019*)
### Fog AI 

- Fogflow: Easy programming of iot services over cloud and edges for smart cities. [[Paper]](https://ieeexplore.ieee.org/document/8022859) [[GitHub]](https://github.com/smartfog/fogflow)
  - Cheng, Bin, Gürkan Solmaz, Flavio Cirillo, Ernö Kovacs, Kazuyuki Terasawa, and Atsushi Kitazawa.
  - IEEE Internet of Things Journal 5, no. 2 (2017): 696-707.

