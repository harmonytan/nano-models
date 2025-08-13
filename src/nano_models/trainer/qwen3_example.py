"""
Qwen3模型训练示例
"""
import torch
import torch.nn as nn
import torch.optim as optim
from qwen3_trainer import Qwen3Trainer, SimpleTokenizer, TextDataset
from ..model.qwen3 import Qwen3Config, Qwen3Transformer


def create_sample_texts():
    """创建示例训练文本"""
    sample_texts = [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "机器学习是人工智能的一个子集，它使用统计学方法让计算机系统能够自动学习和改进，而无需明确编程。",
        "深度学习是机器学习的一个分支，它使用多层神经网络来模拟人脑的学习过程，能够自动学习特征表示。",
        "自然语言处理是人工智能和语言学领域的分支学科，它研究人与计算机之间用自然语言进行有效通信的各种理论和方法。",
        "计算机视觉是人工智能的一个分支，它使计算机能够理解和分析视觉信息，如图像和视频。",
        "强化学习是机器学习的一种方法，它通过与环境交互来学习最优的行为策略。",
        "神经网络是一种模仿生物神经系统的计算模型，由大量相互连接的神经元组成。",
        "卷积神经网络是一种专门用于处理具有网格结构数据的神经网络，如图像数据。",
        "循环神经网络是一种专门用于处理序列数据的神经网络，能够保持状态信息。",
        "Transformer是一种基于注意力机制的神经网络架构，在自然语言处理任务中表现出色。",
        "注意力机制是深度学习中的一种重要技术，它允许模型关注输入数据的不同部分。",
        "预训练模型是在大规模数据上预先训练的模型，可以通过微调适应特定任务。",
        "迁移学习是将在一个任务上学到的知识应用到另一个相关任务上的技术。",
        "数据增强是通过对现有数据进行变换来增加训练数据量的技术。",
        "正则化是防止机器学习模型过拟合的技术，包括L1、L2正则化等。",
        "梯度下降是优化神经网络参数的基本算法，通过计算梯度来更新参数。",
        "反向传播是训练神经网络的核心算法，用于计算梯度。",
        "激活函数是神经网络中的非线性函数，如ReLU、Sigmoid、Tanh等。",
        "批量归一化是一种加速神经网络训练的技术，通过归一化激活值来稳定训练。",
        "Dropout是一种防止过拟合的技术，在训练时随机丢弃一些神经元。"
    ]
    return sample_texts


def main():
    """主函数：演示如何训练Qwen3模型"""
    print("🚀 开始Qwen3模型训练演示...")
    
    # 1. 创建示例文本数据
    print("\n📝 准备训练数据...")
    sample_texts = create_sample_texts()
    print(f"创建了 {len(sample_texts)} 条示例文本")
    
    # 2. 创建分词器
    print("\n🔤 初始化分词器...")
    tokenizer = SimpleTokenizer(vocab_size=1000)
    
    # 3. 创建数据集
    print("\n📊 创建训练数据集...")
    dataset = TextDataset(sample_texts, tokenizer, max_length=128)
    print(f"数据集大小: {len(dataset)}")
    
    # 4. 创建Qwen3模型配置
    print("\n🏗️ 创建Qwen3模型...")
    config = Qwen3Config.from_preset('tiny')  # 使用最小的模型配置进行演示
    print(f"模型配置: {config}")
    
    # 5. 创建Qwen3模型
    model = Qwen3Transformer(config)
    print(f"模型创建完成，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 6. 创建优化器和损失函数
    print("\n⚙️ 设置训练组件...")
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # 7. 创建Qwen3训练器
    print("\n🎯 初始化Qwen3训练器...")
    trainer = Qwen3Trainer(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=2,  # 小批次大小，适合演示
        device="auto",
        save_dir="./qwen3_checkpoints",
        max_length=128
    )
    
    # 8. 开始训练
    print("\n🔥 开始训练...")
    trainer.train(num_epochs=3, save_every=1, log_every=5)
    
    # 9. 评估模型
    print("\n📈 评估模型...")
    eval_loss = trainer.eval()
    
    # 10. 测试文本生成
    print("\n✨ 测试文本生成...")
    test_prompts = [
        "人工智能",
        "机器学习",
        "深度学习"
    ]
    
    for prompt in test_prompts:
        print(f"\n提示: {prompt}")
        generated = trainer.generate_text(prompt, max_length=50, temperature=0.7)
        print(f"生成: {generated}")
    
    # 11. 获取训练统计
    print("\n📊 训练统计信息:")
    stats = trainer.get_training_stats()
    print(f"  当前轮数: {stats['current_epoch']}")
    print(f"  最佳损失: {stats['best_loss']:.4f}")
    print(f"  总训练步数: {stats['total_training_steps']}")
    print(f"  使用设备: {stats['device']}")
    print(f"  词汇表大小: {stats['vocab_size']}")
    print(f"  最大序列长度: {stats['max_length']}")
    
    print("\n🎉 Qwen3模型训练演示完成！")


def test_model_loading():
    """测试模型加载功能"""
    print("\n🔄 测试模型加载功能...")
    
    try:
        # 创建新的模型和训练器
        config = Qwen3Config.from_preset('tiny')
        model = Qwen3Transformer(config)
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 创建简单的数据集
        sample_texts = ["测试文本"]
        tokenizer = SimpleTokenizer(vocab_size=1000)
        dataset = TextDataset(sample_texts, tokenizer, max_length=64)
        
        trainer = Qwen3Trainer(
            model=model,
            dataset=dataset,
            tokenizer=tokenizer,
            optimizer=optimizer,
            criterion=criterion,
            batch_size=1
        )
        
        # 尝试加载最佳模型
        success = trainer.load("qwen3_checkpoints/qwen3_best_model.pt")
        
        if success:
            print("✅ 模型加载成功！")
            
            # 显示训练统计
            stats = trainer.get_training_stats()
            print(f"  当前轮数: {stats['current_epoch']}")
            print(f"  最佳损失: {stats['best_loss']:.4f}")
            
            # 测试推理
            test_input = torch.randint(0, 100, (1, 64))
            with torch.no_grad():
                output = model(test_input)
            print(f"  模型推理正常，输出形状: {output.shape}")
            
        else:
            print("❌ 模型加载失败！")
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")


if __name__ == "__main__":
    main()
    
    # 测试模型加载
    test_model_loading()
