#!/usr/bin/env python
# coding: utf-8

# 中文金融保险意图分类 - 基于 Unsloth + BERT 路线
# 使用 hfl/chinese-macbert-base 作为骨干编码器，通过 AutoModelForSequenceClassification 添加分类头
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
# 数据准备
# ============================================================
# 选项 1：直接从本地 CSV/JSON 加载
#   每条样本需要两个字段：
#     - "text"  : 用户的中文问题字符串
#     - "label" : 对应的意图 id（整数，0 ~ num_labels-1）
#
# from datasets import load_dataset
# dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})
#
# 选项 2：从 HuggingFace Hub 加载已有中文意图数据集，例如：
#   dataset = load_dataset("thu-coai/LCCC", ...)  # 替换为实际数据集
#
# 选项 3（快速验证用）：内联示例数据，仅用于调试流程，请务必替换为真实数据

from datasets import Dataset

sample_data = [
    # 理赔申请 (0)
    {"text": "我要申请理赔，上周发生了交通事故", "label": 0},
    {"text": "怎么提交理赔申请？需要哪些材料", "label": 0},
    {"text": "我的住院费用能报销吗", "label": 0},
    {"text": "理赔审核需要多久", "label": 0},
    {"text": "我想查一下理赔进度", "label": 0},

    # 产品咨询 (1)
    {"text": "重疾险和医疗险有什么区别", "label": 1},
    {"text": "你们有哪些适合老人的保险产品", "label": 1},
    {"text": "这款意外险保障哪些情况", "label": 1},
    {"text": "百万医疗险的免赔额是多少", "label": 1},
    {"text": "分红险的收益是怎么算的", "label": 1},

    # 保费查询 (2)
    {"text": "我的保费是多少，什么时候扣款", "label": 2},
    {"text": "下个月保费要到期了，金额是多少", "label": 2},
    {"text": "为什么这次保费比上次多", "label": 2},
    {"text": "查一下我的年缴保费", "label": 2},
    {"text": "续费金额怎么查", "label": 2},

    # 投保申请 (3)
    {"text": "我想给孩子买一份教育金险", "label": 3},
    {"text": "怎么在线投保重疾险", "label": 3},
    {"text": "我想投保寿险，需要体检吗", "label": 3},
    {"text": "能帮我推荐一款适合我的健康险吗", "label": 3},
    {"text": "我要购买一份意外险", "label": 3},

    # 保单变更 (4)
    {"text": "我要修改受益人信息", "label": 4},
    {"text": "能帮我把保额升级吗", "label": 4},
    {"text": "我想变更缴费方式，从年缴改为月缴", "label": 4},
    {"text": "保单地址需要更新", "label": 4},
    {"text": "联系方式变了，怎么修改", "label": 4},

    # 退保申请 (5)
    {"text": "我想退保，能退多少钱", "label": 5},
    {"text": "退保流程是什么", "label": 5},
    {"text": "我要取消这份保险", "label": 5},
    {"text": "退保会有损失吗", "label": 5},
    {"text": "我要申请退保，请告诉我步骤", "label": 5},

    # 续保服务 (6)
    {"text": "我的保险快到期了，怎么续保", "label": 6},
    {"text": "自动续保是怎么扣款的", "label": 6},
    {"text": "能帮我办理续保吗", "label": 6},
    {"text": "续保后保障内容会变吗", "label": 6},
    {"text": "我要手动续费", "label": 6},

    # 账户管理 (7)
    {"text": "我忘记登录密码了", "label": 7},
    {"text": "怎么修改账户绑定的手机号", "label": 7},
    {"text": "我想查看我的保单列表", "label": 7},
    {"text": "电子保单怎么下载", "label": 7},
    {"text": "账户余额怎么查询", "label": 7},

    # 紧急求助 (8)
    {"text": "我现在出了车祸，需要紧急救援", "label": 8},
    {"text": "客户突发心脏病在医院，如何紧急理赔", "label": 8},
    {"text": "请问24小时紧急救援电话是多少", "label": 8},
    {"text": "我在国外住院，怎么启动境外紧急服务", "label": 8},
    {"text": "紧急救援怎么申请", "label": 8},

    # 投诉建议 (9)
    {"text": "我对理赔结果不满意，想投诉", "label": 9},
    {"text": "你们的客服态度很差，我要投诉", "label": 9},
    {"text": "我有个建议，希望改进你们的App", "label": 9},
    {"text": "上次理赔被无理拒绝，我要投诉", "label": 9},
    {"text": "对保险条款有意见，想反映一下", "label": 9},
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

# Trainer 需要字段名为 "labels"（注意复数）
train_dataset = train_dataset.rename_column("label", "labels")
val_dataset   = val_dataset.rename_column("label", "labels")


# ============================================================
# 类别权重（用于不平衡数据集）
# ============================================================
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

labels_array = train_dataset["labels"]
class_weights = compute_class_weight(
    "balanced",
    classes=np.arange(num_labels),
    y=labels_array,
)
print("类别权重:", class_weights)


# ============================================================
# 评估指标：准确率 + 加权 F1（金融保险场景更关注 F1）
# ============================================================
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1_weighted": f1}


# ============================================================
# 训练配置
# ============================================================
from transformers import TrainingArguments, Trainer
from unsloth import is_bfloat16_supported

training_args = TrainingArguments(
    output_dir = "outputs/chinese_finance_intent",
    num_train_epochs = 5,             # BERT 风格模型通常需要多个 epoch
    per_device_train_batch_size = 32,
    per_device_eval_batch_size  = 32,
    gradient_accumulation_steps = 1,
    warmup_ratio = 0.1,
    learning_rate = 3e-5,             # 中文 BERT 微调推荐 2e-5 ~ 5e-5
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 5,
    optim = "adamw_torch",
    weight_decay = 0.01,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = True,
    metric_for_best_model = "f1_weighted",
    greater_is_better = True,
    seed = 3407,
    report_to = "none",
)

trainer = Trainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = train_dataset,
    eval_dataset  = val_dataset,
    args = training_args,
    compute_metrics = compute_metrics,
)

trainer_stats = trainer.train()


# ============================================================
# 推理验证
# ============================================================
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model = model,
    tokenizer = tokenizer,
    device = 0 if torch.cuda.is_available() else -1,
)

test_sentences = [
    "我要申请理赔，材料已经准备好了",
    "你们有没有适合30岁女性的重疾险",
    "我想退保，请问需要哪些手续",
    "我的密码忘了，怎么找回",
    "对上次的理赔结果不满意，想投诉",
]

print("\n意图分类推理结果：")
for s in test_sentences:
    result = classifier(s)[0]
    print(f"  输入: {s}")
    print(f"  意图: {result['label']}  置信度: {result['score']:.4f}")
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
