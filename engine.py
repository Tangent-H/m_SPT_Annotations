import math
import sys
from typing import Iterable, Optional
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from lib import utils
import random
import numpy as np
import os
from tqdm import tqdm


# All Structured positions
vit_operation_dict = {'q': 0, 'k': 1, 'v': 2, 'proj': 3, 'fc1': 4, 'fc2': 5}


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, scaler=None):
    '''
    这个函数应该是用来计算一个epoch中的待训练权重的梯度的，这里freeze了预训练模型
    '''
    model.train()
    criterion.train()

    # set random seed
    random.seed(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if amp:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)
        else:
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)

        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        elif scaler != 'naive':
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    model=model, create_graph=is_second_order)
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_structured_param_num(structured_type=None, in_dim=768, out_dim=768, low_rank_dim=8):
    if structured_type =='lora':
        return in_dim * low_rank_dim + low_rank_dim * out_dim   #todo 和文章中2*din*dout*rank的结论不符
    elif structured_type =='adapter':
        return out_dim * low_rank_dim + low_rank_dim * out_dim + low_rank_dim + out_dim
    else:
        raise NotImplementedError


def get_sensitivity(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, device: torch.device,
                    amp: bool = True, dataset=None, low_rank_dim=8,
                    structured_vector=True, exp_name=None,
                    structured_type=None, alpha=5., beta=5., last_dim=False,
                    structured_only=False, sensitivity_batch_num=8):

    """Get the sensitivity and the trainable parameter configurations."""
    """输入模型和loss，获得敏感度"""

    # Hyper-parameters alpha and beta, controlling the balance between structured and unstructured tuning
    print(f'Ratio for structually tuning matrices: {alpha}, structurally tuning vectors: {beta}')   #todo 这里的ratio是什么意思
    model.train()
    criterion.train()   # loss function

    # set fixed seed
    random.seed(0)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Getting sensitivity, batch'
    print_freq = 10

    # Sensitivity set S， model.named_parameters()返回一个迭代器，包含模型的所有有名字+参数（state_dict里面的？），先初始化为0
    grad_dict = {name: 0. for name, _ in model.named_parameters()}

    # Accumulating gradient for a epoch
    # Should reach similar results using half of the training samples
    for idx, (samples, targets) in enumerate(data_loader):

        print(f'===== {header}: {idx}')
        if idx >= sensitivity_batch_num:    #默认batch_size=64, batch_num=8，所以默认训练的样本是512
            break

        samples = samples.to(device, non_blocking=True) # 非阻塞函数，不会等待GPU完成才执行接下来的操作
        targets = targets.to(device, non_blocking=True)

        model.zero_grad()   #在执行反向传播之前先清空梯度

        if amp:
            with torch.cuda.amp.autocast(): #with torch.cuda.amp.autocast():块表明在这个作用域内的运算会自动选择合适的精度进行（主要是float16），从而在不牺牲模型精度的前提下提高训练速度和减少内存使用。
                outputs = model(samples)
                loss = criterion(outputs, targets)
        else:
            outputs = model(samples)
            loss = criterion(outputs, targets)  # 输入一个batch的数据和标签，fanhu一个batch的loss（单个数值）

        loss_value = loss.item()    # 转换成python number
        # 如果loss不是有限的（inf或者nan），说明超参数出现了问题，则停止训练
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss.backward()
        for name, param in model.named_parameters():
            grad_dict[name] += (param.grad**2).detach() # grad_dict里面的敏感度不参与反向传播，所以detach出计算图

        torch.cuda.synchronize()    # 用来确保当前进程中的所有CUDA操作完成，包括前面的反向传播和梯度更新
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=0.) # 计算敏感度没有optimize，所以lr=0
    #! 到这里已经计算出了所有的参数的敏感度了
    # Two dicts to keep the partially tuned params by name
    grad_shapes = {}
    grad_shapes_int = {}

    # Pre-defined keywords for calculating sensitivity (kwd: keywords)
    # 带有这些关键字的layer直接fully tune，跳过敏感度
    grad_skip_kwd_list = ['head', 'cls_token', 'patch_embed', 'pos_embed']  # Fully tune head and class token, freeze patch_embed,
                                                                            # we find pos_embed can be either fully ft or unstructured ft, doesn't make much difference
    grad_matrix_kwd_list = ['.q.', '.k.', '.v.', 'proj', 'fc']  # Might structurally tune the matrices: q, k, v, proj, fc1, and fc2
    grad_vector_kwd_list = ['norm', 'bias']  # Might structurly tune the vectors

    for key in grad_dict.keys():
        if not any(kwd in key for kwd in grad_skip_kwd_list):
            grad_shapes[key] = grad_dict[key].shape # 排除掉full tuning的层后保存每个参数名字对应的参数形状
            grad_shapes_int[key] = np.cumprod(list(grad_dict[key].shape))[-1]   # np.cumprod是累积乘积（类比np.cumsum），这里只取最后一个结果，也就是这层的参数的个数，便于后面与长larger_tensor对应

    large_tensor = torch.cat([grad_dict[key].flatten() for key in grad_shapes.keys()])  # 把所有的参数（所有可能微调的层）的敏感度拼接成一个大的tensor

    # Sometimes fewer parameters may have better performance on certain datasets,
    # we get results for several parameter budgets
    # When # of params is less than 0.2, very likely that we are only using unstructured tuning
    param_num_dict = {1.0: 0, 0.8: 0, 0.6: 0, 0.4: 0, 0.2: 0, 0.1: 0, 0.05: 0}  #? 这个字典的作用暂时不知道

    # Sweep for configs matching the budget
    # Actually, simply set param_num to be a precise number,
    # e.g., 0.4, should not give you results that are too far from the parameter budget
    grad_sum_dict = {}

    print('===== Sweeping top-tau sensitive parameters to find ones meeting the target budgets...')
    for param_num in tqdm(range(1, 80)):

        param_num = param_num * 0.02

        # Rank the total sensitivity
        _, indexes = large_tensor.topk(math.ceil(param_num * 1e6))  #? 选取敏感度最高的param_num*1e6个参数，1e6可能是模型的总参数量？

        # Build up masks for unstructured tuning
        tmp_large_tensor = torch.zeros_like(large_tensor, device='cuda')
        tmp_large_tensor[indexes] = 1.  # mask，选中了最敏感k个参数（但是是在一个大的一维tensor里面）

        tmp_large_tensor_list = tmp_large_tensor.split([shape for shape in grad_shapes_int.values()])   # 把这个大的tensor按照每层参数的个数分割成一个list，split是在第0个维度上进行的

        structured_param_num = 0
        structured_names = []   # 结构化调参的参数键名
        tuned_vectors = []

        unstructured_param_num = 0
        unstructured_name_shapes = {}
        unstructured_name_shapes_int = {}
        unstructured_grad_mask = {} # 因为反向传播只调整了部分参数，所以这里保存了调整的参数的mask

        for i, key in enumerate(grad_shapes.keys()):
            # key是剔除掉了full tuning参数的名字，i是参数的索引（需要从tmp_large_tensor_list里面取出对应的参数）
            # 注意tmp_large_tensor_list是一个mask，mask中为1的参数是敏感参数
            grad_sum = tmp_large_tensor_list[i].view(grad_shapes[key]).sum() # 好像view没用，计算该层敏感参数总数
            grad_sum_dict[key] = grad_sum   # 计算这层参数敏感度的总和
            # 如果参数是vectors
            if any(kwd in key for kwd in grad_vector_kwd_list):
                # 如果这一次遍历到的参数名字中包含了vector关键字，那么执行下面的内容(文章中没有提到这点)
                # A trick to also structurally tune vectors when more than 20% of the parameters are sensitive.
                # As the vectors are small, the parameter budget is most likely to be preserved
                # len(grad_shapes[key]) == 1再次验证这个参数层是vector，一个tensor的shape的长度就是这个tensor的维度
                # grad_sum>=...就是敏感参数超过了这个向量的长度的20%，就把这个向量当做structured vector
                #todo: 不过，如何对一个向量做结构化调参？
                if structured_vector and len(grad_shapes[key]) == 1 \
                        and grad_sum >= list(grad_shapes[key])[0] / beta:   # list(grad_shapes[key])[0]返回的就是vector的长度

                    cur_param_num = list(grad_shapes[key])[0]   # cur->current
                    structured_param_num += list(grad_shapes[key])[0]   # 记录结构化参数的数量
                    tuned_vectors.append(key)

                # Unstructured tuning
                else:
                    if not structured_only:

                        cur_param_num = grad_sum.item()

                        unstructured_param_num += grad_sum.item()
                        unstructured_name_shapes[key] = tmp_large_tensor_list[i].view(grad_shapes[key]).shape   # 好像也多此一举了，grad_shapes[key]就是这个参数的shape
                        unstructured_name_shapes_int[key] = np.cumprod(list(grad_dict[key].shape))[-1]
                        unstructured_grad_mask[key] = tmp_large_tensor_list[i].view(grad_shapes[key])   # 这个重要，保存了这个参数的mask，为1的才会反向传播
            # 如果这层参数是矩阵
            elif any(kwd in key for kwd in grad_matrix_kwd_list):
                # 假设要对这层参数进行结构化调参（给定了结构化调参的类型（lora, adaptor)和输入输出维度（采用的是矩阵左乘输入向量）），先看看参数量有多大，再分情况选择是否结构化调参
                cur_structured_param_num = get_structured_param_num(structured_type=structured_type,
                                                             low_rank_dim=low_rank_dim, in_dim=grad_shapes[key][1],
                                                             out_dim=grad_shapes[key][0])

                # Structured
                if grad_sum >= cur_structured_param_num / alpha:    # todo 文章的意思不是grad_sum> cur_structure_param_num就用结构化调参吗？

                    cur_param_num = cur_structured_param_num

                    structured_param_num += cur_structured_param_num
                    structured_names.append(key)

                # Unstructured
                else:
                    if not structured_only:
                        cur_param_num = grad_sum.item()

                        unstructured_param_num += grad_sum
                        unstructured_name_shapes[key] = tmp_large_tensor_list[i].view(grad_shapes[key]).shape
                        unstructured_name_shapes_int[key] = np.cumprod(list(grad_dict[key].shape))[-1]
                        unstructured_grad_mask[key] = tmp_large_tensor_list[i].view(grad_shapes[key])

            else:
                raise NotImplementedError

        # Pre-defined 12 blocks
        # 这6个参数对应的就是q,k,v,o,fc1,fc2
        tuned_matrices = [[0, 0, 0, 0, 0, 0] for _ in range(12)]    # for _ in range(12)就是生成12个[0, 0, 0, 0, 0, 0]

        for name in structured_names:
            attr = name.split('.')

            if len(attr) != 5:
                continue

            block_idx = int(attr[1])
            operation_idx = int(vit_operation_dict[attr[3]])
            tuned_matrices[block_idx][operation_idx] = 1 

        for k in param_num_dict:
            v = param_num_dict[k]
            total_params = (unstructured_param_num + structured_param_num + 768) / 1e6

            # Save the configurations when closer to the target parameter
            if abs(total_params - k) <= abs(v - k):
                param_num_dict[k] = total_params

                res = {'unstructured_name_shapes': unstructured_name_shapes,
                          'unstructured_name_shapes_int': unstructured_name_shapes_int,
                          'params': total_params,
                          'unstructured_params': unstructured_param_num,
                          'structured_params': structured_param_num,
                          'unstructured_indexes': torch.nonzero(torch.cat([unstructured_grad_mask[key].flatten() for key in unstructured_grad_mask.keys()])).squeeze(-1) if unstructured_param_num != 0 else torch.zeros(0).long(),
                          'tuned_matrices': tuned_matrices,
                          'tuned_vectors': tuned_vectors
                        }

                if not os.path.exists('sensitivity_{}/{}'.format(exp_name, dataset)):
                    os.makedirs('sensitivity_{}/{}'.format(exp_name, dataset))
                    print('creating folder: ' + 'sensitivity_{}/{}'.format(exp_name, dataset))

                utils.save_on_master(res, 'sensitivity_{}/{}/param_req_{}.pth'.format(exp_name, dataset, k))
                del res

    print('budgets: real params: ', param_num_dict)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, amp=True):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        try:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        except RuntimeError:
            # class_num <= 5
            acc1 = accuracy(output, target, topk=(1,))
            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1[0].item(), n=batch_size)
            metric_logger.meters['acc5'].update(0., n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
