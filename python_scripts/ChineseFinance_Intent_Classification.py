#!/usr/bin/env python
# coding: utf-8

# 中文金融保险多意图识别 - 基于 Unsloth + BERT 路线
# 模式：multi-label（一句话可同时触发多个意图）
# 使用 hfl/chinese-macbert-base 作为骨干编码器，通过 AutoModelForSequenceClassification 添加分类头
# 损失函数：BCEWithLogitsLoss（每个意图独立二分类）
# 训练框架: Unsloth FastModel + HuggingFace Trainer

# ### 安装依赖
# pip install unsloth transformers datasets scikit-learn accelerate

# ### Unsloth 加载

from unsloth import FastModel
from transformers import AutoModelForSequenceClassification
import torch
import os

os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"

max_seq_length = 128  # 意图分类句子通常较短，128 足够
dtype = None          # 自动检测，Ampere+ 使用 bf16
load_in_4bit = False  # 编码器分类模型不建议 4bit，保持精度

# ============================================================
# 意图标签定义（根据你的业务修改）
# 请确保 id2label 和 label2id 完全对应，且 num_labels 与类别数一致
# ============================================================
id2label = {
    0: "理赔申请",
    1: "产品咨询",
    2: "保费查询",
    3: "投保申请",
    4: "保单变更",
    5: "退保申请",
    6: "续保服务",
    7: "账户管理",
    8: "紧急求助",
    9: "投诉建议",
}

label2id = {v: k for k, v in id2label.items()}

num_labels = len(id2label)

model, tokenizer = FastModel.from_pretrained(
    model_name = "hfl/chinese-macbert-base",   # 中文 BERT，可换为 hfl/chinese-roberta-wwm-ext
    auto_model  = AutoModelForSequenceClassification,
    max_seq_length = max_seq_length,
    dtype = dtype,
    num_labels  = num_labels,
    full_finetuning = True,   # 全量微调，数据少时可改为 False 并用 LoRA
    id2label = id2label,
    label2id = label2id,
    problem_type = "multi_label_classification",  # 多意图：每个标签独立 sigmoid + BCE
    load_in_4bit = load_in_4bit,
)

# 如果显存紧张，可以改用 LoRA（full_finetuning 改为 False，并启用下面的 PEFT）
# model = FastModel.get_peft_model(
#     model,
#     r = 16,
#     target_modules = ["query", "key", "value", "dense"],  # BERT 的注意力层名称
#     lora_alpha = 16,
#     lora_dropout = 0,
#     bias = "none",
#     use_gradient_checkpointing = False,
#     random_state = 3407,
#     task_type = "SEQ_CLS",
# )


# ============================================================
# 数据准备（multi-label 格式）
# ============================================================
# 多意图识别中，每条样本的标签是一个 float 多热向量，长度 = num_labels
# 例如同时触发「理赔申请」和「投诉建议」时：
#   labels = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
#
# 选项 1：从本地 CSV 加载
#   CSV 格式：text 列 + 每个意图一列（0/1），或 text + labels 列（逗号分隔的 id）
#   加载后需调用下面的 encode_labels() 转换为多热向量
#
# 选项 2（快速验证用）：内联示例数据，包含多意图样本

from datasets import Dataset
import numpy as np

def make_multilabel(active_ids: list) -> list:
    """把激活的意图 id 列表转为长度 num_labels 的 float 多热向量"""
    vec = [0.0] * num_labels
    for i in active_ids:
        vec[i] = 1.0
    return vec

sample_data = [
    # 单意图样本
    {"text": "我要申请理赔，上周发生了交通事故",         "labels": make_multilabel([0])},
    {"text": "怎么提交理赔申请？需要哪些材料",           "labels": make_multilabel([0])},
    {"text": "理赔审核需要多久",                        "labels": make_multilabel([0])},
    {"text": "重疾险和医疗险有什么区别",                "labels": make_multilabel([1])},
    {"text": "你们有哪些适合老人的保险产品",             "labels": make_multilabel([1])},
    {"text": "我的保费是多少，什么时候扣款",             "labels": make_multilabel([2])},
    {"text": "为什么这次保费比上次多",                  "labels": make_multilabel([2])},
    {"text": "我想给孩子买一份教育金险",                "labels": make_multilabel([3])},
    {"text": "怎么在线投保重疾险",                     "labels": make_multilabel([3])},
    {"text": "我要修改受益人信息",                     "labels": make_multilabel([4])},
    {"text": "我想变更缴费方式，从年缴改为月缴",         "labels": make_multilabel([4])},
    {"text": "我想退保，能退多少钱",                   "labels": make_multilabel([5])},
    {"text": "退保会有损失吗",                        "labels": make_multilabel([5])},
    {"text": "我的保险快到期了，怎么续保",              "labels": make_multilabel([6])},
    {"text": "续保后保障内容会变吗",                   "labels": make_multilabel([6])},
    {"text": "我忘记登录密码了",                      "labels": make_multilabel([7])},
    {"text": "电子保单怎么下载",                      "labels": make_multilabel([7])},
    {"text": "我现在出了车祸，需要紧急救援",             "labels": make_multilabel([8])},
    {"text": "请问24小时紧急救援电话是多少",            "labels": make_multilabel([8])},
    {"text": "我对理赔结果不满意，想投诉",              "labels": make_multilabel([9])},
    {"text": "你们的客服态度很差，我要投诉",            "labels": make_multilabel([9])},

    # 多意图样本（同时触发多个意图）
    {"text": "我出了车祸，需要紧急救援，同时想申请理赔",  "labels": make_multilabel([0, 8])},
    {"text": "我住院了，能申请理赔吗？需要先了解一下保障范围", "labels": make_multilabel([0, 1])},
    {"text": "我想买重疾险，保费大概多少",              "labels": make_multilabel([1, 2])},
    {"text": "我想买一份健康险，然后把旧保单退掉",       "labels": make_multilabel([3, 5])},
    {"text": "这款产品续保时保费会涨吗",               "labels": make_multilabel([1, 2, 6])},
    {"text": "我要投保，同时把之前的联系方式改一下",     "labels": make_multilabel([3, 4])},
    {"text": "理赔被拒了，我想投诉，同时了解一下重新申请的流程", "labels": make_multilabel([0, 9])},
    {"text": "我要申请退保，退保金额能直接转到账户里吗",  "labels": make_multilabel([5, 7])},
    {"text": "续保的同时帮我把受益人改成我老婆",        "labels": make_multilabel([4, 6])},
    {"text": "出了紧急事故，理赔进度怎么查",            "labels": make_multilabel([0, 8])},
]

raw_dataset = Dataset.from_list(sample_data)

# 80/20 拆分
split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
    )


train_dataset = split_dataset["train"].map(tokenize_function, batched=True)
val_dataset   = split_dataset["test"].map(tokenize_function, batched=True)
# labels 字段已经是 float 多热向量，不需要 rename


# ============================================================
# 每标签正样本权重（multi-label 不平衡处理）
# pos_weight[i] = 负样本数 / 正样本数，传给 BCEWithLogitsLoss
# ============================================================
import numpy as np

labels_matrix = np.array(train_dataset["labels"])  # shape: (N, num_labels)
pos_count = labels_matrix.sum(axis=0).clip(min=1)   # 每列正样本数
neg_count = len(labels_matrix) - pos_count
pos_weight = torch.tensor(neg_count / pos_count, dtype=torch.float32)
print("每意图正样本权重:", {id2label[i]: f"{pos_weight[i]:.2f}" for i in range(num_labels)})


# ============================================================
# 评估指标（multi-label）
# ============================================================
from sklearn.metrics import f1_score, classification_report

THRESHOLD = 0.5  # sigmoid 输出超过此阈值则认为该意图被触发，可调整

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits -> sigmoid -> 二值化
    probs = 1 / (1 + np.exp(-logits))       # sigmoid，shape: (N, num_labels)
    preds = (probs >= THRESHOLD).astype(int)
    labels = np.array(labels).astype(int)

    f1_micro  = f1_score(labels, preds, average="micro",    zero_division=0)
    f1_macro  = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_sample = f1_score(labels, preds, average="samples",  zero_division=0)
    return {
        "f1_micro":   f1_micro,
        "f1_macro":   f1_macro,
        "f1_samples": f1_sample,
    }


# ============================================================
# 训练配置
# ============================================================
from transformers import TrainingArguments, Trainer
from unsloth import is_bfloat16_supported

training_args = TrainingArguments(
    output_dir = "outputs/chinese_finance_intent",
    num_train_epochs = 5,
    per_device_train_batch_size = 16,   # multi-label 样本稍大，batch 适当缩小
    per_device_eval_batch_size  = 16,
    gradient_accumulation_steps = 2,
    warmup_ratio = 0.1,
    learning_rate = 3e-5,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 5,
    optim = "adamw_torch",
    weight_decay = 0.01,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = True,
    metric_for_best_model = "f1_micro",  # multi-label 常用 micro F1 作为主指标
    greater_is_better = True,
    seed = 3407,
    report_to = "none",
)

# 自定义 Trainer：注入 pos_weight 到 BCEWithLogitsLoss
class MultiLabelTrainer(Trainer):
    def __init__(self, pos_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight.to(self.model.device) if pos_weight is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

trainer = MultiLabelTrainer(
    pos_weight = pos_weight,
    model = model,
    processing_class = tokenizer,
    train_dataset = train_dataset,
    eval_dataset  = val_dataset,
    args = training_args,
    compute_metrics = compute_metrics,
)

trainer_stats = trainer.train()


# ============================================================
# 推理验证（multi-label：sigmoid + 阈值，可同时输出多个意图）
# ============================================================
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_intents(text: str, threshold: float = THRESHOLD):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0]          # shape: (num_labels,)
    probs = torch.sigmoid(logits).cpu().numpy()
    results = [
        {"intent": id2label[i], "score": float(probs[i])}
        for i in range(num_labels) if probs[i] >= threshold
    ]
    # 按置信度降序
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

test_sentences = [
    "我要申请理赔，材料已经准备好了",
    "你们有没有适合30岁女性的重疾险，保费大概多少",
    "我出了车祸，需要紧急救援，同时想了解理赔流程",
    "我想退保，退保金可以转到我账户里吗",
    "对上次的理赔结果不满意，想投诉，同时想重新申请",
]

print("\n多意图识别推理结果：")
for s in test_sentences:
    intents = predict_intents(s)
    print(f"  输入: {s}")
    if intents:
        for it in intents:
            print(f"    → {it['intent']}  ({it['score']:.4f})")
    else:
        print(f"    → 未识别到意图（所有意图概率 < {THRESHOLD}）")
    print()


# ============================================================
# 模型保存
# ============================================================
save_dir = "outputs/chinese_finance_intent_final"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"模型已保存到: {save_dir}")

# 如需上传到 HuggingFace Hub:
# model.push_to_hub("your_name/chinese-finance-intent", token="YOUR_HF_TOKEN")
# tokenizer.push_to_hub("your_name/chinese-finance-intent", token="YOUR_HF_TOKEN")
