import torch
import torch.nn as nn
import torch.fft
from einops import rearrange
from model.Linear_extractor import Linear_extractor as expert
from model.RevIN import RevIN
from torch.distributions.normal import Normal
class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input mini-batches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)#都是(224,2),# 按专家索引排序非零门控
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)#各样本对应的专家索引（去重）(224,1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]# # 各专家对应的样本索引
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()# 每个专家接收的样本数
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for an expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            # stitched = stitched.mul(self._nonzero_gates)
            stitched = torch.einsum("i...,ij->i...", stitched, self._nonzero_gates)

        shape = list(expert_out[-1].shape)
        shape[0] = self._gates.size(0)
        zeros = torch.zeros(*shape, requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined#（224,512,1）

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class encoder_u(nn.Module):
    def __init__(self, seq_len, num_experts,hidden_size):
        super(encoder_u, self).__init__()
        input_size = seq_len
        num_experts = num_experts
        encoder_hidden_size = hidden_size

        self.distribution_fit = nn.Sequential(nn.Linear(input_size, encoder_hidden_size, bias=False), nn.ReLU(),
                                              nn.Linear(encoder_hidden_size, num_experts, bias=False))

    def forward(self, x):
        mean = torch.mean(x, dim=-1)
        out = self.distribution_fit(mean)
        return out

class Linear_extractor_cluster(nn.Module):#基于线性的聚类提取器
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, noisy_gating = True,num_experts = 2,seq_len = 105,k = 1, enc_in = 22, CI = 1,d_model = 256,moving_avg =25,hidden_size = 256):
        super(Linear_extractor_cluster, self).__init__()
        self.noisy_gating = noisy_gating#T
        self.num_experts = num_experts#2，专家数量M，即分布簇的数量
        self.input_size = seq_len
        self.k = k#每个样本选择的专家数k（k≤M）
        # instantiate experts
        self.experts = nn.ModuleList([expert(seq_len,d_model,enc_in,CI,moving_avg) for _ in range(self.num_experts)])#线性模式特征提取器选择2个
        self.W_h = nn.Parameter(torch.eye(self.num_experts))#投影矩阵
        self.gate = encoder_u(seq_len, num_experts,hidden_size)## 均值编码器，对应Encoder_μ
        self.noise = encoder_u(seq_len, num_experts,hidden_size)##方差编码器，对应Encoder_σ

        self.n_vars = enc_in#7,变量数
        self.revin = RevIN(self.n_vars)#实例归一化

        self.CI = CI#1
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)#初始化Softplus激活函数，用于将方差参数约束为正数
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    def cv_squared(self, x):#通过最小化变异系数（Coefficient of Variation, CV）的平方，迫使不同专家的使用频率和负载尽可能均匀，避免某些专家过度活跃或闲置。
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.用于计算每个专家在添加噪声后被选入 Top-K 的概率，以实现专家负载均衡
        Args:
        clean_values: a `Tensor` of shape [batch, n].无噪声的门控得分（对应论文中的清洁 logits，即 H(X_{n,:})）
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.注入噪声后的门控得分
        noise_stddev: a `Tensor` of shape [batch, n], or None噪声标准差
        noisy_top_values: a `Tensor` of shape [batch, m].噪声门控得分中前 m（m=k+1）的得分，用于确定阈值
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].输出prob：形状为[batch, num_experts]，表示每个专家在噪声扰动下被选入 Top-k 的概率
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):#在DUET的时间聚类模块中，通过 Noisy Top-K Gating 为每个时间序列动态选择 k 个最相关的 “专家”
        """Noisy top-k gating.
        See paper: https://arxiv.org/abs/1701.06538.
        Args:
          x: input Tensor with shape [batch_size, input_size]
          train: a boolean - we only add noise at training time.
          noise_epsilon: a float
        Returns:
          gates: a Tensor with shape [batch_size, num_experts]
          load: a Tensor with shape [num_experts]
        """
        clean_logits = self.gate(x)#均值

        if self.noisy_gating and train:
            raw_noise_stddev = self.noise(x)#(224,2)
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon#(224,2),Softplus确保非负
            noise = torch.randn_like(clean_logits)#(224,2), 标准正态分布噪声
            noisy_logits = clean_logits + (noise * noise_stddev)#(224,2),注入噪声的门控得分
            logits = noisy_logits @ self.W_h#得到H(n),(224,2)
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)#(224,2),得到的H(x)经过softmax
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)# 选择k+1个最高分，top_indices是选择的专家索引，top_logits是选择的排名前k+1专家的概率
        top_k_logits = top_logits[:, : self.k]#(224,1),保留前k个
        top_k_indices = top_indices[:, : self.k]#挑选前k个专家，并且记录选择的专家索引
        top_k_gates = top_k_logits / (
            top_k_logits.sum(1, keepdim=True) + 1e-6
        )  # normalization,归一化权重

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)#(224,2),稀疏门控矩阵,其中仅前k个专家位置有权重，其余为0

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(0)#tensor（2）load 表示所有专家（Linear-based Pattern Extractor）的负载，元素值越大，代表对应专家被当前批量样本选中的概率或次数越高。
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)#变异系数平方损失
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)## 稀疏分配器
        if self.CI:
            x_norm = rearrange(x, "(x y) l c -> x l (y c)", y=self.n_vars)#（32,512,7）
            x_norm = self.revin(x_norm, "norm")#重复了
            x_norm = rearrange(x_norm, "x l (y c) -> (x y) l c", y=self.n_vars)#（224,512,1）
        else:
            x_norm = self.revin(x, "norm")

        expert_inputs = dispatcher.dispatch(x_norm)#样本分配：通过dispatcher.dispatch将归一化后的样本分配给选中的专家。

        gates = dispatcher.expert_to_gates()
        expert_outputs = [
            self.experts[i](expert_inputs[i]) for i in range(self.num_experts)
        ]#对于选择了不同专家的通道，分别通过线性模型分解时间序列的趋势和季节成分，提取特征，假设总共224个通道，有134个选择了第一个专家，其他的通道选择了第二个专家
        y = dispatcher.combine(expert_outputs)#y:224,512,1

        return y, loss

if __name__ == "__main__":
    # 设置随机种子以便结果可复现
    torch.manual_seed(42)
    x= torch.randn(64,70,22)
    x= x.permute(0, 2, 1).reshape(64 * 22, 70, 1).to('cuda:0')
    # model1 = Mahalanobis_mask(70)
    model2 = Linear_extractor_cluster(noisy_gating = True,num_experts = 3,seq_len = 70,k = 2, enc_in = 22, CI = 1,d_model = 70,moving_avg =25,hidden_size = 70).to('cuda:0')
    # y = model1(x)
    y2 = model2(x)
    print(y2[0].shape)#[64,1,22,22]
    print(y2[0],y2[1])