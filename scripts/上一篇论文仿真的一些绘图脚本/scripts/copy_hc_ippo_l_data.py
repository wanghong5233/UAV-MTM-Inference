#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全复制 hc_ippo_l 算法数据的脚本

该脚本从 experiment_result_sensitivity_plusIPPOL 目录中提取所有 hc_ippo_l 算法的数据，
并将其复制到 experiment_result_sensitivity_plusIPPO 目录中对应的位置，保持目录结构完整。

安全特性:
1. 试运行模式 (--dry-run): 仅显示将要执行的操作，不实际复制
2. 详细日志记录
3. 复制前验证源目录和目标目录的完整性
4. 不删除已存在的数据，而是跳过或提示用户

作者: Assistant
日期: 2025-09-11
"""

import os
import shutil
from pathlib import Path
import logging
import argparse
from datetime import datetime


def setup_logging():
    """
    设置日志配置
    
    Returns:
        logger: 配置好的日志记录器
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('copy_hc_ippo_l_data.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def find_hc_ippo_l_directories(source_root):
    """
    在源目录中查找所有 hc_ippo_l 算法的目录
    
    Args:
        source_root (str): 源根目录路径
        
    Returns:
        list: 包含所有 hc_ippo_l 目录路径的列表
    """
    hc_ippo_l_dirs = []
    source_path = Path(source_root)
    
    if not source_path.exists():
        raise FileNotFoundError(f"源目录不存在: {source_root}")
    
    # 遍历所有参数目录
    for param_dir in source_path.iterdir():
        if param_dir.is_dir():
            # 遍历参数值目录
            for param_value_dir in param_dir.iterdir():
                if param_value_dir.is_dir():
                    # 查找 hc_ippo_l 目录
                    hc_ippo_l_path = param_value_dir / "hc_ippo_l"
                    if hc_ippo_l_path.exists() and hc_ippo_l_path.is_dir():
                        relative_path = hc_ippo_l_path.relative_to(source_path)
                        hc_ippo_l_dirs.append(str(relative_path))
    
    return hc_ippo_l_dirs


def copy_hc_ippo_l_data(source_root, target_root, logger, dry_run=False):
    """
    安全复制 hc_ippo_l 算法数据到目标目录
    
    Args:
        source_root (str): 源根目录路径
        target_root (str): 目标根目录路径
        logger: 日志记录器
        dry_run (bool): 是否为试运行模式
    """
    source_path = Path(source_root)
    target_path = Path(target_root)
    
    # 确保目标根目录存在
    if not target_path.exists():
        raise FileNotFoundError(f"目标目录不存在: {target_root}")
    
    # 查找所有 hc_ippo_l 目录
    hc_ippo_l_dirs = find_hc_ippo_l_directories(source_root)
    
    if not hc_ippo_l_dirs:
        logger.warning("未找到任何 hc_ippo_l 目录")
        return
    
    logger.info(f"找到 {len(hc_ippo_l_dirs)} 个 hc_ippo_l 目录")
    
    if dry_run:
        logger.info("=== 试运行模式：以下操作仅为预览，不会实际执行 ===")
    
    will_copy_count = 0
    will_skip_count = 0
    copied_count = 0
    skipped_count = 0
    error_count = 0
    
    for relative_path in hc_ippo_l_dirs:
        source_dir = source_path / relative_path
        target_dir = target_path / relative_path
        
        # 检查源目录是否存在和完整
        if not source_dir.exists():
            logger.error(f"源目录不存在: {source_dir}")
            error_count += 1
            continue
        
        # 计算源目录中的文件数量
        try:
            source_files = list(source_dir.rglob("*"))
            source_file_count = len([f for f in source_files if f.is_file()])
        except Exception as e:
            logger.error(f"无法读取源目录 {source_dir}: {str(e)}")
            error_count += 1
            continue
        
        # 检查目标目录是否已存在
        if target_dir.exists():
            if dry_run:
                logger.warning(f"[试运行] 目标目录已存在，将跳过: {relative_path}")
                will_skip_count += 1
            else:
                logger.warning(f"目标目录已存在，跳过复制: {relative_path}")
                skipped_count += 1
            continue
        
        if dry_run:
            logger.info(f"[试运行] 将复制 {source_file_count} 个文件: {relative_path}")
            will_copy_count += 1
        else:
            try:
                # 确保目标目录的父目录存在
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                
                # 复制目录
                shutil.copytree(source_dir, target_dir)
                
                # 验证复制结果
                target_files = list(target_dir.rglob("*"))
                target_file_count = len([f for f in target_files if f.is_file()])
                
                if source_file_count == target_file_count:
                    logger.info(f"成功复制 {target_file_count} 个文件: {relative_path}")
                    copied_count += 1
                else:
                    logger.error(f"复制不完整: {relative_path} (源: {source_file_count}, 目标: {target_file_count})")
                    error_count += 1
                
            except Exception as e:
                logger.error(f"复制失败 {relative_path}: {str(e)}")
                error_count += 1
    
    if dry_run:
        logger.info(f"=== 试运行结果预览 ===")
        logger.info(f"将复制: {will_copy_count} 个目录")
        logger.info(f"将跳过: {will_skip_count} 个目录（已存在）")
        logger.info(f"错误: {error_count} 个")
    else:
        logger.info(f"=== 复制完成 ===")
        logger.info(f"成功复制: {copied_count} 个目录")
        logger.info(f"跳过: {skipped_count} 个目录（已存在）")
        logger.info(f"失败: {error_count} 个目录")


def verify_copy_results(target_root, logger):
    """
    验证复制结果
    
    Args:
        target_root (str): 目标根目录路径
        logger: 日志记录器
    """
    target_path = Path(target_root)
    hc_ippo_l_dirs = find_hc_ippo_l_directories(target_root)
    
    logger.info(f"验证结果: 在目标目录中找到 {len(hc_ippo_l_dirs)} 个 hc_ippo_l 目录")
    
    # 检查每个目录的完整性
    for relative_path in hc_ippo_l_dirs:
        dir_path = target_path / relative_path
        seed_dirs = [d for d in dir_path.iterdir() if d.is_dir() and d.name.startswith('seed_')]
        
        logger.info(f"  {relative_path}: {len(seed_dirs)} 个 seed 目录")
        
        # 检查每个 seed 目录是否包含必要的文件
        for seed_dir in seed_dirs:
            config_file = seed_dir / "config_snapshot.json"
            metrics_file = seed_dir / "metrics.csv"
            
            if not config_file.exists():
                logger.warning(f"    缺少文件: {config_file}")
            if not metrics_file.exists():
                logger.warning(f"    缺少文件: {metrics_file}")


def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="安全复制 hc_ippo_l 算法数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python copy_hc_ippo_l_data.py --dry-run    # 试运行模式，仅预览操作
  python copy_hc_ippo_l_data.py              # 实际复制数据
        """
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        help='试运行模式：仅显示将要执行的操作，不实际复制文件'
    )
    parser.add_argument(
        '--source', 
        type=str, 
        help='源目录路径（默认为 experiment_result_sensitivity_plusIPPOL）'
    )
    parser.add_argument(
        '--target', 
        type=str, 
        help='目标目录路径（默认为 experiment_result_sensitivity_plusIPPO）'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging()
    
    # 设置源目录和目标目录
    current_dir = Path(__file__).parent.parent
    source_root = Path(args.source) if args.source else current_dir / "experiment_result_sensitivity_plusIPPOL"
    target_root = Path(args.target) if args.target else current_dir / "experiment_result_sensitivity_plusIPPO"
    
    if args.dry_run:
        logger.info("=== 试运行模式启动 ===")
    else:
        logger.info("=== 开始复制 hc_ippo_l 算法数据 ===")
    
    logger.info(f"源目录: {source_root}")
    logger.info(f"目标目录: {target_root}")
    logger.info(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 验证目录存在性
    if not source_root.exists():
        logger.error(f"源目录不存在: {source_root}")
        return 1
    
    if not target_root.exists():
        logger.error(f"目标目录不存在: {target_root}")
        return 1
    
    try:
        # 执行复制操作
        copy_hc_ippo_l_data(str(source_root), str(target_root), logger, dry_run=args.dry_run)
        
        # 如果不是试运行，则验证复制结果
        if not args.dry_run:
            logger.info("=== 验证复制结果 ===")
            verify_copy_results(str(target_root), logger)
        
        if args.dry_run:
            logger.info("=== 试运行完成 ===")
            logger.info("如需实际执行复制，请运行: python copy_hc_ippo_l_data.py")
        else:
            logger.info("=== hc_ippo_l 数据复制任务完成 ===")
        
        return 0
        
    except Exception as e:
        logger.error(f"复制过程中发生错误: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
