"""
配置管理模块

提供配置文件读取、命令行参数解析和配置验证功能。
支持 YAML 配置文件和命令行参数，实现配置参数优先级处理。
"""

import argparse
import os
import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path

from models.data_models import Config
from models.interfaces import IConfigManager
from models.exceptions import ConfigurationError


class ConfigManager(IConfigManager):
    """配置管理器实现类
    
    负责处理配置文件和命令行参数，提供统一的配置访问接口。
    实现配置参数优先级：命令行参数 > 配置文件 > 默认值
    """
    
    def __init__(self, config_path: Optional[str] = None, args: Optional[List[str]] = None):
        """初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认为 config.yaml
            args: 命令行参数列表，默认使用 sys.argv
        """
        self.config_path = config_path or "config.yaml"
        self.args = args
        self._config: Optional[Config] = None
        self._file_config: Dict[str, Any] = {}
        self._cmd_args: Optional[argparse.Namespace] = None
        
    def load_config(self) -> Config:
        """加载配置信息
        
        按优先级合并配置：命令行参数 > 配置文件 > 默认值
        
        Returns:
            Config: 完整的配置对象
            
        Raises:
            ConfigurationError: 配置加载或验证失败时抛出
        """
        try:
            # 1. 解析命令行参数（先解析以获取可能的配置文件路径）
            self._parse_command_line()
            
            # 2. 加载配置文件（使用可能更新的配置文件路径）
            self._load_config_file()
            
            # 3. 合并配置并创建 Config 对象
            merged_config = self._merge_configurations()
            
            # 4. 验证配置
            self._config = self._create_and_validate_config(merged_config)
            
            return self._config
            
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            else:
                raise ConfigurationError(f"配置加载失败: {str(e)}")
    
    def _load_config_file(self) -> None:
        """加载 YAML 配置文件
        
        Raises:
            ConfigurationError: 配置文件格式错误或读取失败时抛出
        """
        if not os.path.exists(self.config_path):
            # 配置文件不存在时使用空配置，不报错
            self._file_config = {}
            return
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._file_config = yaml.safe_load(f) or {}
                
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"配置文件格式错误: {str(e)}", 
                self.config_path
            )
        except Exception as e:
            raise ConfigurationError(
                f"无法读取配置文件: {str(e)}", 
                self.config_path
            )
    
    def _parse_command_line(self) -> None:
        """解析命令行参数"""
        parser = argparse.ArgumentParser(
            description="视频物体检测系统",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
使用示例:
  python main.py --video input.mp4 --confidence 0.6 --save
  python main.py --video input.mp4 --output output.mp4 --classes person car
  python main.py --config custom_config.yaml --video input.mp4
            """
        )
        
        # 必需参数
        parser.add_argument(
            '--video', '-v',
            type=str,
            help='输入视频文件路径 (.mp4 格式)'
        )
        
        # 可选参数
        parser.add_argument(
            '--output', '-o',
            type=str,
            help='输出视频文件路径'
        )
        
        parser.add_argument(
            '--confidence', '-c',
            type=float,
            help='检测置信度阈值 (0.0-1.0)'
        )
        
        parser.add_argument(
            '--classes',
            nargs='+',
            help='目标检测类别列表'
        )
        
        parser.add_argument(
            '--model',
            type=str,
            help='检测模型文件路径'
        )
        
        parser.add_argument(
            '--config',
            type=str,
            help='配置文件路径'
        )
        
        parser.add_argument(
            '--no-display',
            action='store_true',
            help='不显示实时视频窗口'
        )
        
        parser.add_argument(
            '--save',
            action='store_true',
            help='保存标注后的视频'
        )
        
        parser.add_argument(
            '--max-fps',
            type=int,
            help='最大帧率限制'
        )
        
        parser.add_argument(
            '--log-level',
            type=str,
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            help='日志级别'
        )
        
        parser.add_argument(
            '--log-file',
            type=str,
            help='日志文件路径'
        )
        
        parser.add_argument(
            '--no-console-log',
            action='store_true',
            help='禁用控制台日志输出'
        )
        
        # 解析参数
        if self.args is not None:
            self._cmd_args = parser.parse_args(self.args)
        else:
            self._cmd_args = parser.parse_args()
            
        # 如果指定了配置文件，更新配置文件路径并重新加载
        if self._cmd_args.config:
            self.config_path = self._cmd_args.config
    
    def _merge_configurations(self) -> Dict[str, Any]:
        """合并配置参数
        
        优先级：命令行参数 > 配置文件 > 默认值
        
        Returns:
            Dict[str, Any]: 合并后的配置字典
        """
        # 默认配置
        merged = {
            'video_path': None,
            'output_path': None,
            'confidence_threshold': 0.5,
            'target_classes': None,
            'model_path': None,
            'display_video': True,
            'save_video': False,
            'max_fps': 30,
            'log_level': 'INFO',
            'log_file_path': None,
            'enable_console_log': True,
            'log_max_file_size': 10 * 1024 * 1024,  # 10MB
            'log_backup_count': 5
        }
        
        # 应用配置文件中的设置
        if self._file_config:
            # 处理嵌套的配置结构
            if 'detection' in self._file_config:
                detection_config = self._file_config['detection']
                if 'confidence_threshold' in detection_config:
                    merged['confidence_threshold'] = detection_config['confidence_threshold']
                if 'target_classes' in detection_config:
                    merged['target_classes'] = detection_config['target_classes']
                if 'model_path' in detection_config:
                    merged['model_path'] = detection_config['model_path']
            
            if 'video' in self._file_config:
                video_config = self._file_config['video']
                if 'max_fps' in video_config:
                    merged['max_fps'] = video_config['max_fps']
            
            if 'output' in self._file_config:
                output_config = self._file_config['output']
                if 'save_annotated_video' in output_config:
                    merged['save_video'] = output_config['save_annotated_video']
                if 'output_directory' in output_config:
                    merged['output_path'] = output_config['output_directory']
            
            if 'system' in self._file_config:
                system_config = self._file_config['system']
                if 'display_video' in system_config:
                    merged['display_video'] = system_config['display_video']
                if 'log_level' in system_config:
                    merged['log_level'] = system_config['log_level']
            
            if 'logging' in self._file_config:
                logging_config = self._file_config['logging']
                if 'level' in logging_config:
                    merged['log_level'] = logging_config['level']
                if 'file_path' in logging_config:
                    merged['log_file_path'] = logging_config['file_path']
                if 'enable_console' in logging_config:
                    merged['enable_console_log'] = logging_config['enable_console']
                if 'max_file_size' in logging_config:
                    merged['log_max_file_size'] = logging_config['max_file_size']
                if 'backup_count' in logging_config:
                    merged['log_backup_count'] = logging_config['backup_count']
            
            # 处理顶层配置
            for key in ['video_path', 'output_path', 'confidence_threshold', 
                       'target_classes', 'model_path', 'display_video', 
                       'save_video', 'max_fps', 'log_level', 'log_file_path', 
                       'enable_console_log']:
                if key in self._file_config:
                    merged[key] = self._file_config[key]
        
        # 应用命令行参数（最高优先级）
        if self._cmd_args:
            if self._cmd_args.video:
                merged['video_path'] = self._cmd_args.video
            if self._cmd_args.output:
                merged['output_path'] = self._cmd_args.output
            if self._cmd_args.confidence is not None:
                merged['confidence_threshold'] = self._cmd_args.confidence
            if self._cmd_args.classes:
                merged['target_classes'] = self._cmd_args.classes
            if self._cmd_args.model:
                merged['model_path'] = self._cmd_args.model
            if self._cmd_args.no_display:
                merged['display_video'] = False
            if self._cmd_args.save:
                merged['save_video'] = True
            if self._cmd_args.max_fps is not None:
                merged['max_fps'] = self._cmd_args.max_fps
            if self._cmd_args.log_level:
                merged['log_level'] = self._cmd_args.log_level
            if self._cmd_args.log_file:
                merged['log_file_path'] = self._cmd_args.log_file
            if self._cmd_args.no_console_log:
                merged['enable_console_log'] = False
        
        return merged
    
    def _create_and_validate_config(self, config_dict: Dict[str, Any]) -> Config:
        """创建并验证配置对象
        
        Args:
            config_dict: 配置字典
            
        Returns:
            Config: 验证后的配置对象
            
        Raises:
            ConfigurationError: 配置验证失败时抛出
        """
        # 验证必需参数
        if not config_dict.get('video_path'):
            raise ConfigurationError("缺少必需参数: video_path (视频文件路径)")
        
        # 验证视频文件路径
        video_path = config_dict['video_path']
        if not os.path.exists(video_path):
            raise ConfigurationError(f"视频文件不存在: {video_path}")
        
        # 验证文件格式
        if not video_path.lower().endswith('.mp4'):
            raise ConfigurationError(f"不支持的视频格式: {video_path}，仅支持 .mp4 格式")
        
        # 验证置信度阈值
        confidence = config_dict.get('confidence_threshold', 0.5)
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            raise ConfigurationError(f"置信度阈值必须在 0.0-1.0 之间: {confidence}")
        
        # 验证最大帧率
        max_fps = config_dict.get('max_fps', 30)
        if not isinstance(max_fps, int) or max_fps <= 0:
            raise ConfigurationError(f"最大帧率必须是正整数: {max_fps}")
        
        # 验证输出路径
        output_path = config_dict.get('output_path')
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except Exception as e:
                    raise ConfigurationError(f"无法创建输出目录: {output_dir} - {str(e)}")
        
        # 验证日志级别
        log_level = config_dict.get('log_level', 'INFO')
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if log_level.upper() not in valid_log_levels:
            raise ConfigurationError(f"无效的日志级别: {log_level}，支持的级别: {valid_log_levels}")
        
        # 验证日志文件路径
        log_file_path = config_dict.get('log_file_path')
        if log_file_path:
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except Exception as e:
                    raise ConfigurationError(f"无法创建日志目录: {log_dir} - {str(e)}")
        
        # 创建配置对象
        try:
            return Config(
                video_path=config_dict['video_path'],
                output_path=config_dict.get('output_path'),
                confidence_threshold=float(confidence),
                target_classes=config_dict.get('target_classes'),
                model_path=config_dict.get('model_path'),
                display_video=bool(config_dict.get('display_video', True)),
                save_video=bool(config_dict.get('save_video', False)),
                max_fps=int(max_fps),
                log_level=log_level.upper(),
                log_file_path=log_file_path,
                enable_console_log=bool(config_dict.get('enable_console_log', True))
            )
        except Exception as e:
            raise ConfigurationError(f"配置对象创建失败: {str(e)}")
    
    def get_confidence_threshold(self) -> float:
        """获取置信度阈值
        
        Returns:
            float: 置信度阈值
        """
        if self._config is None:
            self.load_config()
        return self._config.confidence_threshold
    
    def get_target_classes(self) -> Optional[List[str]]:
        """获取目标检测类别
        
        Returns:
            Optional[List[str]]: 目标类别列表，None表示检测所有类别
        """
        if self._config is None:
            self.load_config()
        return self._config.target_classes
    
    def get_output_path(self) -> Optional[str]:
        """获取输出文件路径
        
        Returns:
            Optional[str]: 输出文件路径
        """
        if self._config is None:
            self.load_config()
        return self._config.output_path
    
    def get_model_path(self) -> Optional[str]:
        """获取模型文件路径
        
        Returns:
            Optional[str]: 模型文件路径
        """
        if self._config is None:
            self.load_config()
        return self._config.model_path
    
    def should_display_video(self) -> bool:
        """是否显示实时视频
        
        Returns:
            bool: True表示显示，False表示不显示
        """
        if self._config is None:
            self.load_config()
        return self._config.display_video
    
    def should_save_video(self) -> bool:
        """是否保存标注后的视频
        
        Returns:
            bool: True表示保存，False表示不保存
        """
        if self._config is None:
            self.load_config()
        return self._config.save_video
    
    def get_max_fps(self) -> int:
        """获取最大帧率限制
        
        Returns:
            int: 最大帧率
        """
        if self._config is None:
            self.load_config()
        return self._config.max_fps
    
    def get_config(self) -> Config:
        """获取完整的配置对象
        
        Returns:
            Config: 配置对象
        """
        if self._config is None:
            self.load_config()
        return self._config
    
    def get_log_level(self) -> str:
        """获取日志级别
        
        Returns:
            str: 日志级别
        """
        if self._config is None:
            self.load_config()
        return self._config.log_level
    
    def get_log_file_path(self) -> Optional[str]:
        """获取日志文件路径
        
        Returns:
            Optional[str]: 日志文件路径
        """
        if self._config is None:
            self.load_config()
        return self._config.log_file_path
    
    def should_enable_console_log(self) -> bool:
        """是否启用控制台日志
        
        Returns:
            bool: True表示启用，False表示禁用
        """
        if self._config is None:
            self.load_config()
        return self._config.enable_console_log
    
    def validate_config(self, config: Config) -> bool:
        """验证配置对象的有效性
        
        Args:
            config: 要验证的配置对象
            
        Returns:
            bool: 配置有效返回True
            
        Raises:
            ConfigurationError: 配置无效时抛出
        """
        # 验证视频文件
        if not config.video_path:
            raise ConfigurationError("视频文件路径不能为空")
        
        if not os.path.exists(config.video_path):
            raise ConfigurationError(f"视频文件不存在: {config.video_path}")
        
        # 验证置信度
        if not (0.0 <= config.confidence_threshold <= 1.0):
            raise ConfigurationError(f"置信度阈值必须在 0.0-1.0 之间: {config.confidence_threshold}")
        
        # 验证帧率
        if config.max_fps <= 0:
            raise ConfigurationError(f"最大帧率必须是正数: {config.max_fps}")
        
        return True