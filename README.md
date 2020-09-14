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
* [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529): 相比于BERT提出了两点改进，一个是MASK连续的span，平均长度3.8，一个是提出了SBO目标，就是通过边界词和pos emb来预测span中的单词。
* [Revisiting Pre-Trained Models for Chinese Natural Language Processing](https://arxiv.org/abs/2004.13922): 对中文预训练模型进行了研究，并提出MacBERT。掩码策略是分词后WWM + NG，并且mask的词使用同义词进行替换。


### Finetuning Algorithm
* [Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models](https://arxiv.org/abs/1909.11299): 受到dropout的启发，提出了mixout，区别于dropout将相关weight设置为0，mixout将命中的神经元相关的weight设置为pretrain LM的weight来避免灾难性遗忘的产生，并提升了训练的稳定性。
* [Revisiting Few-sample BERT Fine-tuning](https://arxiv.org/abs/2006.05987): 修正BERTAdam；模型重初始化；增加epoch。
* [Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping](https://arxiv.org/abs/2002.06305): Random Seed的不同会影响参数初始化和data载入顺序并对实验结果的影响非常大。
* [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146): ULMFIT提出了三种finetune算法1) discriminative fine-tuning: 区别于之前的全局唯一一个lr，discriminative fine-tuning对每一个layer使用不同的lr进行训练，lr_prev = lr_curr / 2.6；2) slanted triangular learning: 动态学习率，根据迭代轮数动态的修改lr，起初设置较小的lr找到方向，之后线性增大lr加速收敛，后期线性调小lr；3) gradual unfreezing: unfreeze最后一层训一个epoch，之后unfreeze倒数第二层训一个epoch...
* [Sentence Encoders on STILTs: Supplementary Training on Intermediate Labeled-data Tasks](https://arxiv.org/abs/1811.01088): 将BERT Finetuning分两步走，先在数据量充足的中间数据上Finetune一遍，再在下游继续Finetune，这里主要的应用是在finetune RTE/MRPC/STSB之前，先在MNLI上finetune一下。
* [Multi-Task Deep Neural Networks for Natural Language Understanding](https://arxiv.org/abs/1901.11504): 提出MT-DNN，在BERT的基础上使用MultiTask进行finetune，主要分为4个Task：单句分类，文本相似度，对句分类，相关性排序。通过cross-task data的训练使得pretrained LM能够得到更加通用且泛化性更强的representation。
* [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964): 提出DAPT在Domain Data上继续预训练；TAPT在Task Data上继续预训练。DAPT和TAPT最大的区别在于TAPT是在更小的，但是更task relevant的数据上预训练。


### Document Understanding


### Self Supervised Learning
