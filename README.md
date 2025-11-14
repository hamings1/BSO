# BSO: Binary Spiking Online Optimization Algorithm (ICML 2025 poster)

Yu Liang, Yu Yang, Wenjie Wei, Ammar Belatreche, Shuai Wang, Malu Zhang, Yang Yang

University of Electronic Science and Technology of China, Northumbria University

## Abstract

Binary Spiking Neural Networks (BSNNs) offer promising efficiency advantages for resource-constrained computing. However, their training algorithms often require substantial memory overhead due to latent weights storage and temporal processing requirements. To address this issue, we propose Binary Spiking Online (BSO) optimization algorithm, a novel online training algorithm that significantly reduces training memory. BSO achieves this through two key innovations: time-independent memory requirements and elimination of latent weights storage. BSO directly updates weights through flip signals under the online training framework. These signals are triggered when the product of gradient momentum and weights exceeds a threshold, eliminating the need for latent weights during training. To leverage the inherent temporal dynamics of BSNNs, we further introduce T-BSO, a temporal-aware variant that captures gradient information across time steps for adaptive threshold adjustment. Theoretical analysis establishes convergence guarantees for both BSO and T-BSO, with formal regret bounds characterizing their convergence rates. 
Extensive experiments demonstrate that both BSO and T-BSO achieve superior optimization performance compared to existing training methods for BSNNs.

You can use the train_cifar.sh and train_im.sh file to train BSO.

```
./train_cifar.sh
```

```
./train_im.sh
```

### Citation
If you find this project useful in your research, please consider cite.