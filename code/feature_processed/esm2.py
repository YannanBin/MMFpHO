#!/usr/bin/env python3
"""
ESM2 蛋白质序列特征提取脚本
用于将 FASTA 文件中的蛋白质序列转换为数值嵌入向量

使用方法：
1. 修改 main() 函数中的配置参数
2. 直接运行: python esm2_extract.py
"""

import os
import pickle
import numpy as np
import torch
from Bio import SeqIO


def fasta_to_EsmRep(input_fasta, output_file=None, 
                    pretrained_model_params=None,
                    maxlen=1024,
                    Return=True, 
                    Final_pool=True):
    """
    使用 ESM2 模型提取蛋白质序列特征
    
    参数:
        input_fasta: 输入的 FASTA 文件路径
        output_file: 输出的特征文件路径（pickle 格式）
        pretrained_model_params: 预训练模型路径
        maxlen: 序列最大长度（默认 1024）
        Return: 是否返回特征字典（默认 True）
        Final_pool: 是否对序列维度求平均池化（默认 True）
    
    返回:
        output_dict: 序列ID到特征向量的字典
    """
    import esm
    
    # 氨基酸字典
    aa_dict = {amino_acid: i for i, amino_acid in enumerate("ACDEFGHIKLMNPQRSTVWYX")}
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    print("正在加载 ESM2 模型...")
    try:
        if pretrained_model_params and os.path.exists(pretrained_model_params):
            input_embedding_net, alphabet = esm.pretrained.load_model_and_alphabet_local(
                pretrained_model_params
            )
            print(f"已从本地加载模型: {pretrained_model_params}")
        else:
            raise FileNotFoundError("本地模型文件不存在")
    except Exception as e:
        print(f"跳过加载本地预训练模型: {e}")
        print("尝试从 hub 下载 ESM2-650M 模型...")
        
        if pretrained_model_params:
            weight_path = os.path.dirname(os.path.abspath(pretrained_model_params))
            if os.path.exists(weight_path) and os.path.isdir(weight_path):
                torch.hub.set_dir(weight_path)
                print(f"模型将下载到: {weight_path}")
        else:
            print("模型将下载到默认缓存目录")
        
        input_embedding_net, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    
    # 初始化批处理转换器
    batch_converter = alphabet.get_batch_converter()
    input_embedding_net = input_embedding_net.to(device)
    input_embedding_net.eval()
    
    # 提取特征
    output_dict = {}
    real_maxlen = max(1, maxlen - 2)  # 减去 BOS 和 EOS token
    num_layer = len(input_embedding_net.layers)
    
    print(f"开始处理 FASTA 文件: {input_fasta}")
    print(f"序列最大长度: {real_maxlen}")
    print(f"最终池化: {Final_pool}")
    
    seq_count = 0
    for record in SeqIO.parse(open(input_fasta), 'fasta'):
        seq_count += 1
        sequence = str(record.seq[:real_maxlen])
        
        # 将非标准氨基酸替换为 'X'
        sequence = "".join([x if x in aa_dict else 'X' for x in sequence])
        
        # 准备数据
        data = [("protein1", sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        
        # 提取特征
        with torch.no_grad():
            results = input_embedding_net(
                batch_tokens, 
                repr_layers=[num_layer], 
                return_contacts=True
            )
        
        # 获取 token 表示
        token_representations = results["representations"][num_layer]
        embedding = np.array(
            token_representations.squeeze(0).detach().cpu()
        ).astype(np.float16)
        
        # 截取实际序列长度
        embedding = embedding[:real_maxlen + 2, :]
        
        # 是否进行平均池化
        if Final_pool:
            embedding = embedding.mean(0)
        
        output_dict[record.id] = embedding
        
        if seq_count % 10 == 0:
            print(f"已处理 {seq_count} 条序列...")
    
    print(f"总共处理 {seq_count} 条序列")
    
    # 保存结果
    if output_file:
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'wb') as f:
                pickle.dump(output_dict, f)
            print(f'特征已保存至: {output_file}')
        except Exception as e:
            print(f'保存失败: {e}')
            # 尝试保存到临时文件
            import uuid
            tmp_name = str(uuid.uuid4()) + '_esm.pkl'
            tmp_dir = os.path.dirname(input_fasta) if os.path.dirname(input_fasta) else '.'
            output_file = os.path.join(tmp_dir, tmp_name)
            with open(output_file, 'wb') as f:
                pickle.dump(output_dict, f)
            print(f'临时文件已保存至: {output_file}')
    
    if Return:
        return output_dict


def main():
    """
    主函数 - 直接在这里配置参数运行
    """
    
    # ============ 配置参数（在这里修改） ============
    
    # 输入 FASTA 文件路径
    input_fasta = "/data5/heyuhang/mywork/anlifenxi/enzyme.fasta"
    
    # 输出特征文件路径
    output_file = "/data5/heyuhang/mywork/anlifenxi/ph_feature/esm2.pkl"
    
    # 预训练模型路径（可选，None 则从 hub 下载）
    pretrained_model_params = '/data5/heyuhang/mywork/CLEF/pretrained_model/esm2_t33_650M_UR50D.pt'
    # 或者设置为 None 自动下载：
    # pretrained_model_params = None
    
    # 序列最大长度
    maxlen = 1024
    
    # 是否进行平均池化（True: 输出 (1280,), False: 输出 (seq_len, 1280)）
    Final_pool = False
    
    # ============================================
    
    # 检查输入文件
    if not os.path.exists(input_fasta):
        print(f"错误: 输入文件不存在 - {input_fasta}")
        print("请修改 main() 函数中的 input_fasta 路径")
        return
    
    # 配置参数
    esm_config = {
        'input_fasta': input_fasta,
        'output_file': output_file,
        'pretrained_model_params': pretrained_model_params,
        'maxlen': maxlen,
        'Return': False,
        'Final_pool': Final_pool
    }
    
    # 运行特征提取
    print("=" * 60)
    print("ESM2 特征提取开始")
    print("=" * 60)
    print(f"输入文件: {input_fasta}")
    print(f"输出文件: {output_file}")
    print(f"最大长度: {maxlen}")
    print(f"平均池化: {Final_pool}")
    print("=" * 60)
    
    fasta_to_EsmRep(**esm_config)
    
    print("=" * 60)
    print("特征提取完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()