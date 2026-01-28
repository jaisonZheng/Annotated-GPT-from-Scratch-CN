# Annotated-GPT-from-Scratch-CN

想要手搓复现一个Transformer？复现GPT得预训练部分？快去看Andrej Karpathy的神级视频[Let's build GPT: from scratch, in code, spelled out.](https://youtu.be/kCc8FmEb1nY?si=ggbZ0AJ2oHnFjHbP)

需要有详细的中文注释的代码？这个仓库Annotated-GPT-from-Scratch-CN将完美满足你的要求！
仓库中所有代码都是一个字符一个字符手敲的，在详细的中文注释中写了我对代码的理解。

code文件夹下，有各个阶段模型的代码。
从只根据上一个token进行预测的BigramModel，到增加单头注意力，多头注意力，残差连接，Layer Normalization，并最终Scaling Up。我将不同的组件在不同的文件中逐步添加，以展示增加不同组件后的效果。

在diff文件夹中，展示了各个文件之间的差异，如果你只想看变化部分，可以点击下面表格中的超链接进行跳转。

| 起始版本 | 目标版本 | Diff 文件                                                                                                    | 说明                     |
| ---- | ---- | ---------------------------------------------------------------------------------------------------------- | ---------------------- |
| v0   | v1   | [v0_v1.diff](https://github.com/jaisonZheng/Annotated-GPT-from-Scratch-CN/blob/main/difference/v0_v1.diff) | 初版 Bigram 模型到单头注意力的改动  |
| v1   | v2   | [v1_v2.diff](https://github.com/jaisonZheng/Annotated-GPT-from-Scratch-CN/blob/main/difference/v1_v2.diff) | 单头注意力扩展为多头注意力          |
| v2   | v3   | [v2_v3.diff](https://github.com/jaisonZheng/Annotated-GPT-from-Scratch-CN/blob/main/difference/v2_v3.diff) | 引入残差连接与层堆叠             |
| v3   | v4   | [v3_v4.diff](https://github.com/jaisonZheng/Annotated-GPT-from-Scratch-CN/blob/main/difference/v3_v4.diff) | 添加 Layer Normalization |
| v4   | v5   | [v4_v5.diff](https://github.com/jaisonZheng/Annotated-GPT-from-Scratch-CN/blob/main/difference/v4_v5.diff) | 扩充模型规模并引入正则化           |

### 使用方式
你可以用Andrej Karpathy推荐的方式，看完代码之后将代码关掉，然后自己手搓复现一遍。
也可以只是读懂代码，然后将每个文件运行一遍。
注意：v5_Scaling_up.py需要在较好的显卡上运行，其余代码均可以在笔记本电脑上运行。

### 效果预览
#### v0_Bigram模型效果
_未经训练的输出；loss:4.5_
![](IMG-20260127121443731.png)
_经过训练的输出；loss:2.5_
![](IMG-20260127121550420.png)
可以看到，只根据上一个token进行预测的Bigram模型，经过训练之后，似乎也有一点莎士比亚的样子，但是说的明显不是英语。


#### v1_Single-Head_Attention效果
![](IMG-20260127173720488.png)
此时loss ~ 2.4，比2.5稍好，但是生成的文本仍然不是英语。

#### v2_Multi-Head_Attention效果
![](IMG-20260127210506553.png)
_loss ~ 2.3，似乎长进不多，但能看到一些英语单词了。_

#### v3_Residual_Connections
一开始我只是堆叠了Block，以下是效果：
![](IMG-20260127212512074.png)
_loss ~ 2.3，几乎没有长进。甚至可以说有退步(2.29 -> 2.31)，这是因为深层神经网络不好训练_
加了residual connection
![](IMG-20260127215226826.png)_loss大幅度降到了2.1。WOW！_

#### v4_Layer_Normalization
![](IMG-20260127231334200.png)
_loss稍微下降，但是输出仍然不是非常英语，看来我要使出最后的大招：Scaling了！_

#### v5_Scaling_Up
![](IMG-20260128103826230.png)
_loss从2.1大幅下降到了1.5！WOW！！！！_

ok，那让我们来欣赏一下这段文本吧：（经由Gemini3 Flash翻译）

---
**卡普莱特夫人：** 什么！什么羞耻，一个证明？

**朱丽叶：** 什么费马，表演者。卡米洛至高无上！

**奶妈：** 管它是她打雷你把我卖了吗？

**雅赫特：** 哎，先生，奶妈，先生。一个妻子，塞隆，一个井： 如果你的女人yout，谁有事‘这是危险。

**布鲁图斯：** 我只是这背叛了你的孩子们。

**朱丽叶：** 说他听过关于，那残忍的爱 他在我的睡眠中tartituoey于你的。 看，正如知道那些罪恶，我如何听，重要性！

**奥菲迪乌斯：** 那不算。多么经常croops再次，我且 已经告诉 Aymend一个

---

hhhhh，这什么玩意？算了，我再让Gemini3 Flash润色一下好了：

---
**卡普莱特夫人：** 放肆！这是何等羞辱，难道还需更多明证？

**朱丽叶：** 那是何等良驹，何等英姿。唯有卡米洛才是最高贵的骑士！

**奶妈：** 老天爷，难道任由她雷霆大作，就要把我这把老骨头扫地出门吗？

**雅赫特：** 哎，大人，老妈子，大人。一个贤妻，一座堡垒，一眼清泉： 若你们女人的心思飘忽不定，那危险便近在咫尺。

**布鲁图斯：** 我所做的这一切，不过是背叛了你们的孩子。

**朱丽叶：** 且听他分辩那残忍的爱， 他在我梦境中纠缠，夺走你的安宁。 看啊，既然深知这些罪孽，我该如何倾听这所谓的重任！

**奥菲迪乌斯：** 那都无关紧要。这种痛苦又要发作多少次？ 我已经告诉过艾蒙德了……

---

哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈，大功告成！