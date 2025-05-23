**Physical Imaging Model-Guided Deep Variational Despeckling Framework for 
Ultrasound Images**

Despeckling is essential for enhancing the clinical interpretability of ultrasound (US) images, as speckle 
noise can obscure tissue details and complicate diagnoses. Traditional despeckling methods, which rely on physical 
models of US imaging, have proven effective but often suffer from parameter sensitivity, low computational 
efficiency, and a tendency to over-smooth images. In contrast, deep learning (DL) approaches excel at learning from 
large datasets and can significantly reduce speckle noise with high efficiency and flexibility. However, DL methods 
face two key challenges: the scarcity of ground-truth clean US images, which requires the use of simulated clean 
images that might not accurately reflect real-world conditions, and the lack of integration with prior knowledge, 
leading to reduced interpretability and generalizability. In this paper, we propose a novel deep Variational US Image 
Despeckling (VUID) framework that addresses these limitations. VUID integrates the strengths of model-based 
methods by incorporating prior knowledge of US imaging physics and statistical distributions of relevant parameters 
into a variational DL architecture. Unlike previous methods, VUID treats simulated clean US images as the mode 
of the distribution of ground-truth clean images, serving as a guide rather than as absolute ground truth for training, 
which thereby enables a more accurate representation. The framework employs two distinct deep convolutional 
networks to predict the parameters of variational posterior distributions for speckle noise variance and ground-truth 
clean images. These networks are jointly optimized using a hybrid loss function that combines an Evidence Lower 
Bound (ELBO) loss with an image reconstruction loss guided by the US imaging model. We conducted extensive 
experiments comparing VUID with state-of-the-art traditional and DL despeckling methods. Our tests included 
assessing overall despeckling performance on three synthetic and two real US datasets, evaluating robustness and 
generalizability on two unseen real US datasets, and demonstrating clinical value through diagnostic quality 
assessments by medical experts. The results demonstrate that VUID surpasses existing methods across multiple 
metrics, highlighting its potential to improve the diagnostic accuracy and clinical utility of ultrasonic imaging.
