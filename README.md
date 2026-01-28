# Annotated-GPT-from-Scratch-CN

想要手搓复现一个Transformer？复现GPT得预训练部分？快去看Andrej Karpathy的神级视频[Let's build GPT: from scratch, in code, spelled out.](https://youtu.be/kCc8FmEb1nY?si=ggbZ0AJ2oHnFjHbP)

想亲手复现一个 Transformer？想实现 GPT 的预训练部分？强烈推荐你去看 Andrej Karpathy 的神级视频：Let's build GPT: from scratch, in code, spelled out.

需要带有详细中文注释的代码？本仓库 Annotated-GPT-from-Scratch-CN 正好能满足你的需求！
仓库中的所有代码都是我一行一行手敲出来的，并在注释中详细写下了我的理解和思考。

在 code 文件夹下，你可以找到各个阶段模型的代码。
从最基础的 BigramModel（只根据上一个 token 预测下一个），到加入单头注意力、多头注意力、残差连接、Layer Normalization，最后实现 Scaling Up。我将不同的组件分阶段、分文件逐步添加，方便你观察每一步的变化和效果。

在 diff 文件夹中，展示了各个版本之间的差异。如果你只想关注每一步的变化，可以直接点击下表中的超链接跳转查看。

| 起始版本 | 目标版本 | Diff 文件                                                                                                    | 说明                     |
| ---- | ---- | ---------------------------------------------------------------------------------------------------------- | ---------------------- |
| v0   | v1   | [v0_v1.diff](https://github.com/jaisonZheng/Annotated-GPT-from-Scratch-CN/blob/main/difference/v0_v1.diff) | 初版 Bigram 模型到单头注意力的改动  |
| v1   | v2   | [v1_v2.diff](https://github.com/jaisonZheng/Annotated-GPT-from-Scratch-CN/blob/main/difference/v1_v2.diff) | 单头注意力扩展为多头注意力          |
| v2   | v3   | [v2_v3.diff](https://github.com/jaisonZheng/Annotated-GPT-from-Scratch-CN/blob/main/difference/v2_v3.diff) | 引入残差连接与层堆叠             |
| v3   | v4   | [v3_v4.diff](https://github.com/jaisonZheng/Annotated-GPT-from-Scratch-CN/blob/main/difference/v3_v4.diff) | 添加 Layer Normalization |
| v4   | v5   | [v4_v5.diff](https://github.com/jaisonZheng/Annotated-GPT-from-Scratch-CN/blob/main/difference/v4_v5.diff) | 扩充模型规模并引入正则化           |

### 使用方式
你可以按照 Karpathy 推荐的方法，先看懂代码，然后关掉代码，自己手搓一遍加深理解。
也可以直接阅读并运行每个文件，逐步体会模型的演变。
注意：v5_Scaling_up.py 需要较好的显卡支持，其余代码在普通笔记本上也能运行。

### 效果预览
#### v0_Bigram模型效果
_未经训练的输出；loss:4.5_

<img width="696" height="258" alt="IMG-20260127121443731" src="https://github.com/user-attachments/assets/13625864-e77d-4ec5-90b6-c0224eba2007" />


_经过训练的输出；loss:2.5_

<img width="699" height="391" alt="IMG-20260127121550420" src="https://github.com/user-attachments/assets/83d83ef2-c30a-425d-8803-d6c49fb894c5" />

可以看到，只根据上一个token进行预测的Bigram模型，经过训练之后，似乎也有一点莎士比亚的样子，但是说的明显不是英语。


#### v1_Single-Head_Attention效果
<img width="628" height="377" alt="IMG-20260127173720488" src="https://github.com/user-attachments/assets/eed8f581-1de8-46d9-a555-d12b8cb29684" />

此时loss ~ 2.4，比2.5稍好，但是生成的文本仍然不是英语。

#### v2_Multi-Head_Attention效果
<img width="800" height="400" alt="IMG-20260127210506553" src="https://github.com/user-attachments/assets/00907841-a61f-4411-925d-72bd49c7e2c3" />

_loss ~ 2.3，似乎长进不多，但能看到一些英语单词了。_

#### v3_Residual_Connections
一开始我只是堆叠了Block，以下是效果：

<img width="800" height="270" alt="IMG-20260127212512074" src="https://github.com/user-attachments/assets/e52f19c9-0f41-40c7-baf9-0bf28384318b" />

_loss ~ 2.3，几乎没有长进。甚至可以说有退步(2.29 -> 2.31)，这是因为深层神经网络不好训练_

加了residual connection后：

<img width="794" height="397" alt="IMG-20260127215226826" src="https://github.com/user-attachments/assets/55242306-30fc-4ec5-8e8e-8d9fe47457ef" />

_loss大幅度降到了2.1。WOW！_

#### v4_Layer_Normalization
<img width="800" height="284" alt="IMG-20260127231334200" src="https://github.com/user-attachments/assets/eb17a27c-422c-4261-9e6b-b0d428196eb1" />

_loss稍微下降，但是输出仍然不是非常英语，看来我要使出最后的大招：Scaling了！_

#### v5_Scaling_Up
<img width="646" height="515" alt="IMG-20260128103826230" src="https://github.com/user-attachments/assets/0f70c688-99b7-4570-a2a0-fd8c0b53c0a0" />

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
