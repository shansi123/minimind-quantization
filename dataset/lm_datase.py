from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):  # add_system_ratio插入system prompt的概率
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations  # 判断，如果顶层字典中含有tools，则直接返回对话，避免把工具调用相关的的数据弄乱

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]  # 系统提示词，目的是数据加强，使模型在训练时看到更多的带身份设定的对话样本（让模型更加符合设定）
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:  # 判断随机数是否命中概率，如果是，则添加系统提示词
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations  # 构造上下文，随机注入一条系统提示词
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):  # 预训练数据集类（对所有token都要进行训练）
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer  # 分词器，为后面分词做准备
        self.max_length = max_length  # 数据样本长度上限
        self.samples = load_dataset('json', data_files=data_path, split='train')  # 加载json数据

    def __len__(self):
        return len(self.samples)
    # 对[]索引进行重写
    def __getitem__(self, index):
        sample = self.samples[index]  # 引入单条样本
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids  # 使用分词器将text切分，便于后面处理（多余部分直接截断）
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]  # 为切分后的数据标记开始和结束俩个序列边界（训练阶段：标记序列边界）（生成阶段：作为停止条件之一）
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))  # 补全数据，为了训练时的批次化，张量必须统一长度
        input_ids = torch.tensor(input_ids, dtype=torch.long)  # 张量化
        labels = input_ids.clone()  #克隆一个标签，用于后续计算交叉熵损失
        labels[input_ids == self.tokenizer.pad_token_id] = -100  # 将pad_token_id标注为-100，表示不计算损失
        return input_ids, labels


class SFTDataset(Dataset):  # 监督微调数据集类（仅训练assistant的回复部分）
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:  # 处理每一条对话
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]  # 若tools是json字符串则处理为json字典在赋值，是json字典，就直接赋值
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(  # 转化为对话的特殊格式
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)  # 可能有多个bos和eos
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])  # 概率添加系统提示词
        prompt = self.create_chat_prompt(conversations)  # 创建上下文
        prompt = post_processing_chat(prompt)  # 概率去除空白标签
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]  # 限制数据样本长度
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))  # 补全
        labels = self.generate_labels(input_ids)  # 生成对应的标签
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):  # 直接偏好优化
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):  # 生成是否参与损失的掩码
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):  # 人工智能反馈强化学习
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)  # 概率添加系统提示词（工具调用则直接返回）
        use_thinking = random.random() < self.thinking_ratio  # 概率启用
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ""
        }

class AgentRLDataset(Dataset):  # AgentRLDataset 的作用是把“Agent 任务的数据”整理成一个更容易给强化学习 / 轨迹学习使用的结构
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}

"""
SFTDataset：更像“监督学习”，直接让模型学回复
DPODataset：更像“偏好学习”，给出 chosen/rejected 对
RLAIFDataset：更像“生成 prompt，后续再用奖励信号训练”
AgentRLDataset：更像“给 agent 任务准备上下文和目标输出”，适合工具调用、动作生成、轨迹学习这类场景
"""

if __name__ == "__main__":
    pass