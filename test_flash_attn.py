import torch
import os
import inspect

print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
print("CUDA版本:", torch.version.cuda)

try:
    # 尝试导入flash-attn
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    
    print("\nflash-attn导入成功!")
    
    # 创建一些测试数据
    batch_size = 2
    seq_len = 10
    num_heads = 4
    head_dim = 16
    
    # 创建查询、键、值张量 - 使用fp16数据类型
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    
    # 尝试运行flash_attn_func
    try:
        output = flash_attn_func(q, k, v, causal=True)
        print("flash_attn_func运行成功!")
    except Exception as e:
        print("flash_attn_func运行失败:", str(e))
    
    # 测试unpad_input和pad_input
    try:
        attention_mask = torch.ones(batch_size, seq_len, device="cuda")
        hidden_states = torch.randn(batch_size, seq_len, num_heads * head_dim, device="cuda", dtype=torch.float16)
        
        # 检查unpad_input的返回值
        result = unpad_input(hidden_states, attention_mask)
        print(f"unpad_input返回值类型: {type(result)}, 长度: {len(result) if isinstance(result, tuple) else 'N/A'}")
        
        # 正确处理5个返回值
        unpaded_hidden_states, indices, cu_seqlens, max_seqlen, batch_seqlens = result
        print(f"返回值详情:")
        print(f"- unpaded_hidden_states形状: {unpaded_hidden_states.shape}")
        print(f"- indices长度: {len(indices)}")
        print(f"- cu_seqlens: {cu_seqlens}")
        print(f"- max_seqlen: {max_seqlen}")
        print(f"- batch_seqlens: {batch_seqlens}")
            
        # 使用pad_input函数将未填充的隐藏状态填充回原始形状
        padded_hidden_states = pad_input(unpaded_hidden_states, indices, batch_size, seq_len)
        print(f"padded_hidden_states形状: {padded_hidden_states.shape}")
        
        # 验证填充后的形状是否与原始形状相同
        if padded_hidden_states.shape == hidden_states.shape:
            print("形状匹配，unpad_input和pad_input运行成功!")
        else:
            print(f"形状不匹配: 原始={hidden_states.shape}, 填充后={padded_hidden_states.shape}")
    except Exception as e:
        print("unpad_input和pad_input运行失败:", str(e))
        import traceback
        traceback.print_exc()
    
    # 检查flash_attn_varlen_func的参数
    print("\n检查flash_attn_varlen_func的参数:")
    print(inspect.signature(flash_attn_varlen_func))
    
    # 测试flash_attn_varlen_func
    try:
        # 创建不同长度的序列
        # 创建累积序列长度数组
        cu_seqlens_q = torch.tensor([0, 8, 15], device="cuda", dtype=torch.int32)  # 第一个序列长度8，第二个序列长度7
        cu_seqlens_k = torch.tensor([0, 8, 15], device="cuda", dtype=torch.int32)  # 键值对应的序列长度
        max_seqlen_q = 8
        max_seqlen_k = 8
        
        # 创建总长度为15的查询、键、值张量
        q_unpad = torch.randn(15, num_heads, head_dim, device="cuda", dtype=torch.float16)
        k_unpad = torch.randn(15, num_heads, head_dim, device="cuda", dtype=torch.float16)
        v_unpad = torch.randn(15, num_heads, head_dim, device="cuda", dtype=torch.float16)
        
        # 根据实际参数调用函数
        output_unpad = flash_attn_varlen_func(
            q_unpad, k_unpad, v_unpad, 
            cu_seqlens_q=cu_seqlens_q, 
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=True
        )
        print("flash_attn_varlen_func运行成功!")
        print(f"输出形状: {output_unpad.shape}")
    except Exception as e:
        print("flash_attn_varlen_func运行失败:", str(e))
        import traceback
        traceback.print_exc()
        
    # 测试在实际模型中的使用场景
    print("\n测试在实际模型中的使用场景:")
    try:
        # 1. 创建带有注意力掩码的批次数据
        batch_size = 2
        seq_lens = [8, 6]  # 两个序列，长度分别为8和6
        max_seq_len = max(seq_lens)
        
        # 创建注意力掩码
        attention_mask = torch.zeros(batch_size, max_seq_len, device="cuda")
        for i, seq_len in enumerate(seq_lens):
            attention_mask[i, :seq_len] = 1
        
        # 创建隐藏状态
        hidden_dim = num_heads * head_dim
        hidden_states = torch.randn(batch_size, max_seq_len, hidden_dim, device="cuda", dtype=torch.float16)
        
        # 2. 将隐藏状态重塑为多头格式
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, seq_len, num_heads, head_dim)
        
        # 3. 使用unpad_input移除填充
        q = k = v = hidden_states
        q_unpad, indices, cu_seqlens, max_seqlen, batch_seqlens = unpad_input(q, attention_mask)
        k_unpad = v_unpad = q_unpad
        
        # 4. 使用flash_attn_varlen_func
        output_unpad = flash_attn_varlen_func(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True
        )
        
        # 5. 使用pad_input恢复原始形状
        output = pad_input(output_unpad, indices, batch_size, max_seq_len)
        
        print("完整流程测试成功!")
        print(f"输入形状: {hidden_states.shape}")
        print(f"输出形状: {output.shape}")
        
    except Exception as e:
        print("完整流程测试失败:", str(e))
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print("\nflash-attn导入失败:", str(e))
    print("\n检查是否已安装flash-attn:")
    try:
        import subprocess
        result = subprocess.run(["pip", "list"], capture_output=True, text=True)
        packages = result.stdout.split("\n")
        flash_attn_packages = [p for p in packages if "flash" in p.lower()]
        if flash_attn_packages:
            print("找到以下flash相关包:")
            for p in flash_attn_packages:
                print(p)
        else:
            print("未找到flash-attn相关包")
    except Exception as e:
        print("检查安装包时出错:", str(e)) 