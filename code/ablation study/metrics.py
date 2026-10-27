"""
评估指标计算模块
包含回归指标和分类指标的计算函数
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, matthews_corrcoef,
    f1_score, roc_auc_score, roc_curve
)
from scipy.stats import pearsonr, spearmanr


def continuous_to_class(values):
    """
    将连续pH值转换为5类
    
    Args:
        values: numpy array, 连续的pH值
        
    Returns:
        numpy array, 分类标签 (0-4)
    """
    classes = np.zeros(len(values), dtype=int)
    classes[values <= 5.5] = 0
    classes[(values > 5.5) & (values <= 6.5)] = 1
    classes[(values > 6.5) & (values <= 7.5)] = 2
    classes[(values > 7.5) & (values <= 8.5)] = 3
    classes[values > 8.5] = 4
    return classes


def calculate_regression_metrics(y_true, y_pred):
    """
    计算回归指标
    
    Args:
        y_true: numpy array, 真实值
        y_pred: numpy array, 预测值
        
    Returns:
        dict: 包含RMSE, R2, Pearson, Spearman的字典
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pearson_corr, pearson_pvalue = pearsonr(y_true, y_pred)
    spearman_corr, spearman_pvalue = spearmanr(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'R2': r2,
        'Pearson': pearson_corr,
        'Pearson_pvalue': pearson_pvalue,
        'Spearman': spearman_corr,
        'Spearman_pvalue': spearman_pvalue
    }


def calculate_binary_metrics_ovr(y_true_classes, y_pred_classes, num_classes=5):
    """
    One-vs-Rest方式计算二分类指标
    对每个类别分别计算二分类指标，最后取平均
    
    Args:
        y_true_classes: numpy array, 真实的类别标签
        y_pred_classes: numpy array, 预测的类别标签
        num_classes: int, 类别数量，默认5
        
    Returns:
        dict: 包含Accuracy, MCC, F1, AUC的平均值和每个类别的详细值
    """
    accuracies = []
    mccs = []
    f1s = []
    aucs = []
    
    class_details = {}
    
    for target_class in range(num_classes):
        # 将当前类别视为正类，其他类别视为负类
        y_true_binary = (y_true_classes == target_class).astype(int)
        y_pred_binary = (y_pred_classes == target_class).astype(int)
        
        # 计算指标
        acc = accuracy_score(y_true_binary, y_pred_binary)
        mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # 计算AUC（使用预测标签）
        if len(np.unique(y_true_binary)) > 1:
            auc = roc_auc_score(y_true_binary, y_pred_binary)
        else:
            auc = 0.0
        
        accuracies.append(acc)
        mccs.append(mcc)
        f1s.append(f1)
        aucs.append(auc)
        
        # 保存每个类别的详细信息
        class_details[f'Class_{target_class}'] = {
            'Accuracy': acc,
            'MCC': mcc,
            'F1': f1,
            'AUC': auc
        }
    
    return {
        'Accuracy': np.mean(accuracies),
        'MCC': np.mean(mccs),
        'F1': np.mean(f1s),
        'AUC': np.mean(aucs),
        'class_details': class_details
    }


def calculate_multiclass_metrics(y_true_classes, y_pred_classes):
    """
    计算多分类指标
    
    Args:
        y_true_classes: numpy array, 真实的类别标签
        y_pred_classes: numpy array, 预测的类别标签
        
    Returns:
        dict: 包含多分类的准确率、宏平均和微平均的F1和MCC
    """
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    
    acc = accuracy_score(y_true_classes, y_pred_classes)
    f1_macro = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
    f1_micro = f1_score(y_true_classes, y_pred_classes, average='micro', zero_division=0)
    f1_weighted = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_true_classes, y_pred_classes)
    
    return {
        'Multiclass_Accuracy': acc,
        'Multiclass_F1_Macro': f1_macro,
        'Multiclass_F1_Micro': f1_micro,
        'Multiclass_F1_Weighted': f1_weighted,
        'Multiclass_MCC': mcc
    }


def calculate_all_metrics(y_true, y_pred):
    """
    计算所有指标（回归+分类）
    
    Args:
        y_true: numpy array, 真实的连续值
        y_pred: numpy array, 预测的连续值
        
    Returns:
        dict: 包含所有指标的字典
    """
    # 回归指标
    reg_metrics = calculate_regression_metrics(y_true, y_pred)
    
    # 转换为分类
    pred_classes = continuous_to_class(y_pred)
    true_classes = continuous_to_class(y_true)
    
    # 二分类指标（OvR平均）
    binary_metrics = calculate_binary_metrics_ovr(true_classes, pred_classes)
    
    # 多分类指标
    multiclass_metrics = calculate_multiclass_metrics(true_classes, pred_classes)
    
    # 合并所有指标
    all_metrics = {
        **reg_metrics,
        **{k: v for k, v in binary_metrics.items() if k != 'class_details'},
        **multiclass_metrics
    }
    
    # 单独返回类别详细信息
    all_metrics['binary_class_details'] = binary_metrics['class_details']
    
    return all_metrics


def print_metrics(metrics, title="Metrics"):
    """
    格式化打印指标
    
    Args:
        metrics: dict, 指标字典
        title: str, 标题
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    print(f"\n{'回归指标':^30}")
    print(f"{'-'*60}")
    reg_keys = ['RMSE', 'R2', 'Pearson', 'Spearman']
    for key in reg_keys:
        if key in metrics:
            print(f"  {key:.<25} {metrics[key]:>10.4f}")
    
    print(f"\n{'二分类指标 (OvR平均)':^30}")
    print(f"{'-'*60}")
    binary_keys = ['Accuracy', 'MCC', 'F1', 'AUC']
    for key in binary_keys:
        if key in metrics:
            print(f"  {key:.<25} {metrics[key]:>10.4f}")
    
    print(f"\n{'多分类指标':^30}")
    print(f"{'-'*60}")
    multi_keys = ['Multiclass_Accuracy', 'Multiclass_F1_Macro', 'Multiclass_MCC']
    for key in multi_keys:
        if key in metrics:
            print(f"  {key:.<25} {metrics[key]:>10.4f}")
    
    # 打印每个类别的详细信息
    if 'binary_class_details' in metrics:
        print(f"\n{'各类别二分类详细指标':^30}")
        print(f"{'-'*60}")
        for class_name, class_metrics in metrics['binary_class_details'].items():
            print(f"\n  {class_name}:")
            for metric_name, value in class_metrics.items():
                print(f"    {metric_name:.<21} {value:>10.4f}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # 示例用法
    print("评估指标计算模块 - 使用示例\n")
    
    # 生成示例数据
    np.random.seed(42)
    y_true = np.random.uniform(4, 10, 100)
    y_pred = y_true + np.random.normal(0, 0.5, 100)
    
    # 计算所有指标
    metrics = calculate_all_metrics(y_true, y_pred)
    
    # 打印指标
    print_metrics(metrics, "示例：pH预测评估指标")
    
    # 也可以单独计算和打印某一类指标
    print("\n" + "="*60)
    print("单独使用各函数的示例:")
    print("="*60)
    
    reg_metrics = calculate_regression_metrics(y_true, y_pred)
    print("\n回归指标:")
    for k, v in reg_metrics.items():
        print(f"  {k}: {v:.4f}")






def calculate_binary_metrics_ovr(y_true_classes, y_pred_classes, num_classes=5):
    """
    One-vs-Rest方式计算二分类指标
    对每个类别分别计算二分类指标，最后取平均
    安全处理类别样本极少或预测全0的情况
    
    Args:
        y_true_classes: numpy array, 真实的类别标签
        y_pred_classes: numpy array, 预测的类别标签
        num_classes: int, 类别数量，默认5
        
    Returns:
        dict: 包含Accuracy, MCC, F1, AUC的平均值和每个类别的详细值
    """
    accuracies = []
    mccs = []
    f1s = []
    aucs = []
    
    class_details = {}
    
    for target_class in range(num_classes):
        y_true_binary = (y_true_classes == target_class).astype(int)
        y_pred_binary = (y_pred_classes == target_class).astype(int)
        
        # 判断正负样本是否存在
        num_pos = y_true_binary.sum()
        num_neg = len(y_true_binary) - num_pos
        
        if num_pos == 0 or num_neg == 0:
            # 如果类别样本为0或全预测为0，安全返回0.0
            acc = 0.0
            mcc = 0.0
            f1 = 0.0
            auc = 0.0
        else:
            acc = accuracy_score(y_true_binary, y_pred_binary)
            try:
                mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
            except:
                mcc = 0.0
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            try:
                auc = roc_auc_score(y_true_binary, y_pred_binary)
            except:
                auc = 0.0
        
        accuracies.append(acc)
        mccs.append(mcc)
        f1s.append(f1)
        aucs.append(auc)
        
        class_details[f'Class_{target_class}'] = {
            'Accuracy': acc,
            'MCC': mcc,
            'F1': f1,
            'AUC': auc
        }
    
    # 平均值
    mean_acc = np.mean(accuracies)
    mean_mcc = np.mean(mccs)
    mean_f1 = np.mean(f1s)
    mean_auc = np.mean(aucs)
    
    return {
        'Accuracy': mean_acc,
        'MCC': mean_mcc,
        'F1': mean_f1,
        'AUC': mean_auc,
        'class_details': class_details
    }
