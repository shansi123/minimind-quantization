import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 自动分配一个可随反向传播更新、初始为 1 的权重向量，方便后续直接更新和被优化

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # 均方根归一化

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)  # 转化为x的数据类型

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0  # 频率就是这么设计的
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))  # 反解波长公式：求维度索引
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)  # 两个边界索引
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)  # 将low的位置置为0位置处，缩放比例，将high缩放到1位置处，截断（0，1）位置（线性斜坡（Ramp）函数）
            freqs = freqs * (1 - ramp + ramp / factor)  # 频率缩放公式
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # 做外积，形成位置->旋转频率表
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor  # 计算cosm*freqs
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor  # 计算sinm*freqs
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):  # q/k：[batch, num_heads, seq_len, head_dim]  - cos / sin:[batch, seq_len, head_dim]
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)  # 返回后半部分取反然后拼接前半段（原因：RoPE 本质上是对“独立的二维子空间”做旋转。只要我们把d维向量切分成 d/2d/2 个独立的二维平面，至于哪些维度分到同一组，是完全自由的。）
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)  # 求q的词嵌入向量与旋转位置编码的乘积（求带有位置信息的词嵌入向量）
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)  # 求k的词嵌入向量与旋转位置编码的乘积（求带有位置信息的词嵌入向量）
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    # 将kv头集合分别传入，然后根据n_rep值分别复制（分组查询注意力（Grouped-Query Attention））
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))
    
class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)  # [batch_size,seq_len,hidden_size]吗，这里的写法是为了提高代码可读性和解耦与安全
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # [batch_size,seq_len,hidden_size]
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # [batch_size,seq_len,hidden_size]
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)  # 改形状
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # 后面要取反
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)  # 先打上位置编码
        if past_key_value is not None:  # 检查是否有kv缓存
            xk = torch.cat([past_key_value[0], xk], dim=1)  # 有则在把新k存入缓存中（拼接历史k）
            xv = torch.cat([past_key_value[1], xv], dim=1)  # 有则在把新v存入缓存中（拼接历史v）
        past_kv = (xk, xv) if use_cache else None  # 判断要不要使用缓存的kv头
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))  # 这里采用了gqa分组请求头注意力机制，实现了多q头共享统一kv头，节省显存，提高解码吞吐（计算能力），几乎不损失质量，训练代价小
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)  # 构建因果注意力掩码
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9  # 处理padding问题，防止无用计算
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)  # 将形状改为标准的[batch_size, seq_len, hidden_size]
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)  # 门控网络
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)  # 降维，使得向量回归原始形状，后续要计算和映射词表
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)  # 升维,为后续计算留足空间，让向量携带了更多相关数据（如语法之类的）
        self.act_fn = ACT2FN[config.hidden_act]  # 激活函数swish（silu激活）

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))  #先计算门控然后激活，接着与升维数据相乘得到要保留的数据（swiglu激活函数）

class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)  # 路由器（Gate）
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])  # 创建并用pytorch自动管理多个专家（feednetwork）
        self.act_fn = ACT2FN[config.hidden_act]  # 使用swish激活（silu）

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)  # 计算选择专家的概率，为后续选择专家做准备
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)  # 选出每个token的top—k专家，分别得到权重和索引（token都找出最适合的专家）
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)  # 对选中专家的权重进行归一化，是为了扩大差距，使得权重选择更加方便
        y = torch.zeros_like(x_flat)  # 创建一个和x_flat形状一样的空白张量，用于存储后续的权重累加和
        for i, expert in enumerate(self.experts):  # 循环为每一个token选择最好的专家
            mask = (topk_idx == i)  # 判断
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()  # 得到该专家适合解决的token索引（一维张量）
                weight = topk_weight[mask].view(-1, 1)  # 从路由权重中取出对应的分数
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))  # 将各专家对该token的计算结果权重累加后，按索引累加到y张量上
            elif self.training:  # 因为pythorch内部对参数进行反向传播更新时，没有被选中的专家参数为None会报错，所以采用强行构造了一个零乘以该专家所有参数总和的表达式，使得梯度为0，既不影响DDP也不影响y值
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:  # router_aux_loss_coef：辅助均衡损失的权重系数
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)  # 将每个token选中的专家进行独热编码，取平均得到每一个专家的使用概率（一维张量）
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef  # 将load和每一个专家使用概率相乘求和得到损失（数学直觉：如果一个专家既被频繁选中（load大），又获得了很高的平均概率（scores.mean大），那么乘积就大，损失就大。模型在反向传播时，会调整路由器的权重，压低这种“垄断”现象，鼓励各个专家被相对均匀地使用。）（当各个专家被选到的次数越平均时，损失越小）
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()  # 创建一个1维值为0的张量，然后去掉所有为1的维度，变成0维张量
        return y.view(batch_size, seq_len, hidden_dim)

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)  # 自注意力机制计算
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 对输入进行层归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 对输出进行层归一化
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)  # 是否使用混合专家模型

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states  # 存储原始输入张量
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )  # 计算出残差和  返回kv cache
        hidden_states += residual  # 残差连接
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))  # 残差连接+前馈网络或混合专家模型
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config  # 配置
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers  # 词表大小，注意力机制头数量
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # 词嵌入层
        self.dropout = nn.Dropout(config.dropout)  # 随机失活，确保某个特别突出的特征不会对后续操作造成影响
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])  # 解码器层（多头注意力）
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 层归一化
        # 求对应的旋转频率表
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)  # 登记这俩个到pytorch内部，但不到模型权重内部，不进行反向传播时更新
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)  # 登记

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0  # 得到已缓存的序列长度，为后续获取相应旋转位置编码做准备
        hidden_states = self.dropout(self.embed_tokens(input_ids))  # 记录前随机失活一些特征，为了保证不被特殊特征打扰
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            # 改变freqs_cos和freqs_sin的存储位置到hidden_size位置，为了防止attention中出现device mismatch（设备位置错误）
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])  # 获取当前位置的旋转位置编码，为attention中的词嵌入向量注入位置信息做准备
        presents = []
        # 执行多头注意力机制，获得hidden_size和kv cache
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)  # 将kv缓存写入，节省显存，节约计算时间
        hidden_states = self.norm(hidden_states)  # 层归一化
        # 将上述计算过后的aux_loss进行相加（在使用了moefeednetwork的情况下），得到辅助损失，便于对专家的选取进行优化（通过损失惩罚的方式）（梯度是损失函数对参数的偏导）
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()  # 分别左移右移一位对齐如下所示
            """
            [[”我“，”爱“，”你“],
             [”爱“，”你“，”们“]]
            """
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)  # logits->(2, 9, vocab_size)  labels->(2, 9)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)  # 复制出多条生成序列（批量）
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None  # 处理padding
        past_key_values = kwargs.pop("past_key_values", None)  # 取出kv头缓存
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0  # 已缓存的k头数量（token数量）
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature  # 找最后一行用温度系数缩放概率
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)  # 重复惩罚(CTRL 论文提出),对已出现过的 token 降低其 logit 分数,减少重复生成。
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')  # None 等价于 torch.unsqueeze(-1),在末尾加一个维度。原因是为了广播形状匹配，将除了前k个的值都变为-inf
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p  # 逐步累加概率
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0  # top_p采样中对mask的右移一位并补0操作
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)  # 根据概率随机采样或者贪心采样
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)  # torch.where(condition, x, y):condition 为真取 x,为假取 y。已结束的序列被冻结在eos状态
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids