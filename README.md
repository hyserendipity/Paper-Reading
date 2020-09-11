# Paper-Reading


### Adversarial Training
* [Fast is better than free: Revisiting adversarial training](https://arxiv.org/abs/2001.03994): 在FGSM算法前加入Uniform的扰动可以将FGSM算法的效率提升到和PGD相当的效果。
* [Adversarial Training for Free!](https://arxiv.org/abs/1904.12843): 使用PGD内层循环产生的梯度来更新参数以达到Free的效果。
* [SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization](https://arxiv.org/abs/1911.03437): 使用对称的KL散度来计算扰动，并对参数值计算滑动平均，以KL散度/square loss来保证参数值不会剧烈变化。
* [FreeLB: Enhanced Adversarial Training for Natural Language Understanding](https://arxiv.org/abs/1909.11764): 对PGD内层循环里的梯度计算平均值，使用平均值来更新参数以达到Free的效果。
* [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572): 提出FGSM算法。
* [Adversarial Training for Large Neural Language Models](https://arxiv.org/abs/2004.08994): 将对抗训练引入Pretain领域，提出ALUM。
* [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083): 提出PGD算法。
* [Adversarial Training Methods for Semi-Supervised Text Classification](https://arxiv.org/abs/1605.07725): 提出在embedding上加入扰动生成对抗样本。
* [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://arxiv.org/abs/1901.11196): EDA模型使用了四种简单的文本数据增强方案但取得了很不错的效果：SR同义词替换，RI随机插入，RS随机互换，RD随机删除。
* [TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP](https://arxiv.org/abs/2005.05909): 提出了一个用于自然语言处理模型的对抗攻击框架TextAttack：https://github.com/QData/TextAttack
* [Semantically Equivalent Adversarial Rules for Debugging NLP Models](https://www.aclweb.org/anthology/P18-1079/): 设计一系列的规则来生成对抗样本。


### Pretrained Language Model
* [StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding](https://arxiv.org/abs/1908.04577): 在BERT的任务上新加入两个任务word structural objective和sentence structural objective将句子的时序信息引入到预训练当中去。
* [Contextual Embeddings: When Are They Worth It?](https://arxiv.org/abs/2005.09117): 对contextual embedding和word2vec进行了比较，结论是在训练集充足且语句结构简单的任务上word2vec相较BERT效果甚至更好；在语句结构复杂，一词多义较多，unseen words较多的任务上，BERT效果往往更好。
* [ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/abs/1904.09223): 在预训练的时候mask实体以引入先验知识。
* [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942): NSP-> SOP; 减小embedding层维数；对内层的bert layer重复使用。

### Fintuning Algorithm


### Document Understanding


### Self Supervised Learning
