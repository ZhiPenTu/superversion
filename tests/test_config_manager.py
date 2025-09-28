"""
配置管理器单元测试

测试 ConfigManager 类的各种功能，包括配置文件解析、命令行参数处理、
默认值处理和错误处理等。
"""

import pytest
import tempfile
import os
import yaml
from unittest.mock import patch, mock_open
from pathlib import Path

from utils.config_manager import ConfigManager
from models.data_models import Config
from models.exceptions import ConfigurationError


class TestConfigManager:
    """ConfigManager 类的单元测试"""
    
    def setup_method(self):
        """每个测试方法执行前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_video_path = os.path.join(self.temp_dir, "test_video.mp4")
        self.test_config_path = os.path.join(self.temp_dir, "test_config.yaml")
        
        # 创建测试视频文件（空文件用于测试）
        with open(self.test_video_path, 'w') as f:
            f.write("")
    
    def teardown_method(self):
        """每个测试方法执行后的清理"""
        # 清理临时文件
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_config_file(self, config_data: dict):
        """创建测试配置文件
        
        Args:
            config_data: 配置数据字典
        """
        with open(self.test_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
    
    def test_init_with_default_config_path(self):
        """测试使用默认配置文件路径初始化"""
        manager = ConfigManager()
        assert manager.config_path == "config.yaml"
        assert manager.args is None
    
    def test_init_with_custom_config_path(self):
        """测试使用自定义配置文件路径初始化"""
        custom_path = "custom_config.yaml"
        manager = ConfigManager(config_path=custom_path)
        assert manager.config_path == custom_path
    
    def test_init_with_custom_args(self):
        """测试使用自定义命令行参数初始化"""
        custom_args = ["--video", "test.mp4"]
        manager = ConfigManager(args=custom_args)
        assert manager.args == custom_args
    
    def test_load_config_file_success(self):
        """测试成功加载配置文件"""
        # 创建测试配置文件
        config_data = {
            'detection': {
                'confidence_threshold': 0.7,
                'target_classes': ['person', 'car']
            },
            'video': {
                'max_fps': 25
            }
        }
        self.create_test_config_file(config_data)
        
        manager = ConfigManager(config_path=self.test_config_path)
        manager._load_config_file()
        
        assert manager._file_config == config_data
    
    def test_load_config_file_not_exists(self):
        """测试配置文件不存在的情况"""
        non_existent_path = os.path.join(self.temp_dir, "non_existent.yaml")
        manager = ConfigManager(config_path=non_existent_path)
        manager._load_config_file()
        
        # 配置文件不存在时应该使用空配置
        assert manager._file_config == {}
    
    def test_load_config_file_invalid_yaml(self):
        """测试无效的 YAML 配置文件"""
        # 创建无效的 YAML 文件
        with open(self.test_config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        manager = ConfigManager(config_path=self.test_config_path)
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager._load_config_file()
        
        assert "配置文件格式错误" in str(exc_info.value)
    
    def test_parse_command_line_basic_args(self):
        """测试基本命令行参数解析"""
        args = ["--video", self.test_video_path, "--confidence", "0.8"]
        manager = ConfigManager(args=args)
        manager._parse_command_line()
        
        assert manager._cmd_args.video == self.test_video_path
        assert manager._cmd_args.confidence == 0.8
    
    def test_parse_command_line_all_args(self):
        """测试所有命令行参数解析"""
        output_path = os.path.join(self.temp_dir, "output.mp4")
        args = [
            "--video", self.test_video_path,
            "--output", output_path,
            "--confidence", "0.6",
            "--classes", "person", "car", "bicycle",
            "--model", "yolo.pt",
            "--no-display",
            "--save",
            "--max-fps", "20"
        ]
        
        manager = ConfigManager(args=args)
        manager._parse_command_line()
        
        assert manager._cmd_args.video == self.test_video_path
        assert manager._cmd_args.output == output_path
        assert manager._cmd_args.confidence == 0.6
        assert manager._cmd_args.classes == ["person", "car", "bicycle"]
        assert manager._cmd_args.model == "yolo.pt"
        assert manager._cmd_args.no_display is True
        assert manager._cmd_args.save is True
        assert manager._cmd_args.max_fps == 20
    
    def test_merge_configurations_defaults_only(self):
        """测试仅使用默认配置"""
        manager = ConfigManager()
        manager._file_config = {}
        manager._cmd_args = None
        
        merged = manager._merge_configurations()
        
        expected = {
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
        
        assert merged == expected
    
    def test_merge_configurations_file_config(self):
        """测试配置文件优先级"""
        manager = ConfigManager()
        manager._file_config = {
            'detection': {
                'confidence_threshold': 0.7,
                'target_classes': ['person']
            },
            'video': {
                'max_fps': 25
            },
            'output': {
                'save_annotated_video': True
            }
        }
        manager._cmd_args = None
        
        merged = manager._merge_configurations()
        
        assert merged['confidence_threshold'] == 0.7
        assert merged['target_classes'] == ['person']
        assert merged['max_fps'] == 25
        assert merged['save_video'] is True
    
    def test_merge_configurations_cmd_args_priority(self):
        """测试命令行参数最高优先级"""
        args = ["--video", self.test_video_path, "--confidence", "0.9"]
        manager = ConfigManager(args=args)
        manager._file_config = {
            'detection': {
                'confidence_threshold': 0.7
            }
        }
        manager._parse_command_line()
        
        merged = manager._merge_configurations()
        
        # 命令行参数应该覆盖配置文件
        assert merged['video_path'] == self.test_video_path
        assert merged['confidence_threshold'] == 0.9
    
    def test_create_and_validate_config_success(self):
        """测试成功创建和验证配置"""
        config_dict = {
            'video_path': self.test_video_path,
            'output_path': None,
            'confidence_threshold': 0.6,
            'target_classes': ['person'],
            'model_path': None,
            'display_video': True,
            'save_video': False,
            'max_fps': 30
        }
        
        manager = ConfigManager()
        config = manager._create_and_validate_config(config_dict)
        
        assert isinstance(config, Config)
        assert config.video_path == self.test_video_path
        assert config.confidence_threshold == 0.6
        assert config.target_classes == ['person']
        assert config.max_fps == 30
    
    def test_create_and_validate_config_missing_video_path(self):
        """测试缺少视频路径的配置验证"""
        config_dict = {
            'video_path': None,
            'confidence_threshold': 0.5
        }
        
        manager = ConfigManager()
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager._create_and_validate_config(config_dict)
        
        assert "缺少必需参数: video_path" in str(exc_info.value)
    
    def test_create_and_validate_config_video_not_exists(self):
        """测试视频文件不存在的配置验证"""
        non_existent_video = os.path.join(self.temp_dir, "non_existent.mp4")
        config_dict = {
            'video_path': non_existent_video,
            'confidence_threshold': 0.5
        }
        
        manager = ConfigManager()
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager._create_and_validate_config(config_dict)
        
        assert "视频文件不存在" in str(exc_info.value)
    
    def test_create_and_validate_config_invalid_format(self):
        """测试不支持的视频格式验证"""
        invalid_video = os.path.join(self.temp_dir, "test.avi")
        with open(invalid_video, 'w') as f:
            f.write("")
        
        config_dict = {
            'video_path': invalid_video,
            'confidence_threshold': 0.5
        }
        
        manager = ConfigManager()
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager._create_and_validate_config(config_dict)
        
        assert "不支持的视频格式" in str(exc_info.value)
    
    def test_create_and_validate_config_invalid_confidence(self):
        """测试无效置信度阈值验证"""
        config_dict = {
            'video_path': self.test_video_path,
            'confidence_threshold': 1.5  # 超出范围
        }
        
        manager = ConfigManager()
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager._create_and_validate_config(config_dict)
        
        assert "置信度阈值必须在 0.0-1.0 之间" in str(exc_info.value)
    
    def test_create_and_validate_config_invalid_fps(self):
        """测试无效帧率验证"""
        config_dict = {
            'video_path': self.test_video_path,
            'confidence_threshold': 0.5,
            'max_fps': -10  # 负数
        }
        
        manager = ConfigManager()
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager._create_and_validate_config(config_dict)
        
        assert "最大帧率必须是正整数" in str(exc_info.value)
    
    def test_load_config_integration(self):
        """测试完整的配置加载流程"""
        # 创建配置文件
        config_data = {
            'detection': {
                'confidence_threshold': 0.7
            }
        }
        self.create_test_config_file(config_data)
        
        # 命令行参数
        args = ["--video", self.test_video_path, "--save"]
        
        manager = ConfigManager(config_path=self.test_config_path, args=args)
        config = manager.load_config()
        
        # 验证配置合并结果
        assert config.video_path == self.test_video_path
        assert config.confidence_threshold == 0.7  # 来自配置文件
        assert config.save_video is True  # 来自命令行
        assert config.display_video is True  # 默认值
    
    def test_get_confidence_threshold(self):
        """测试获取置信度阈值"""
        args = ["--video", self.test_video_path, "--confidence", "0.8"]
        manager = ConfigManager(args=args)
        
        threshold = manager.get_confidence_threshold()
        assert threshold == 0.8
    
    def test_get_target_classes(self):
        """测试获取目标类别"""
        args = ["--video", self.test_video_path, "--classes", "person", "car"]
        manager = ConfigManager(args=args)
        
        classes = manager.get_target_classes()
        assert classes == ["person", "car"]
    
    def test_get_target_classes_none(self):
        """测试获取目标类别为空的情况"""
        non_existent_config = os.path.join(self.temp_dir, "non_existent.yaml")
        args = ["--video", self.test_video_path]
        manager = ConfigManager(config_path=non_existent_config, args=args)
        
        classes = manager.get_target_classes()
        assert classes is None
    
    def test_get_output_path(self):
        """测试获取输出路径"""
        output_path = os.path.join(self.temp_dir, "output.mp4")
        args = ["--video", self.test_video_path, "--output", output_path]
        manager = ConfigManager(args=args)
        
        path = manager.get_output_path()
        assert path == output_path
    
    def test_should_display_video(self):
        """测试是否显示视频"""
        args = ["--video", self.test_video_path, "--no-display"]
        manager = ConfigManager(args=args)
        
        should_display = manager.should_display_video()
        assert should_display is False
    
    def test_should_save_video(self):
        """测试是否保存视频"""
        args = ["--video", self.test_video_path, "--save"]
        manager = ConfigManager(args=args)
        
        should_save = manager.should_save_video()
        assert should_save is True
    
    def test_get_max_fps(self):
        """测试获取最大帧率"""
        args = ["--video", self.test_video_path, "--max-fps", "25"]
        manager = ConfigManager(args=args)
        
        max_fps = manager.get_max_fps()
        assert max_fps == 25
    
    def test_validate_config_success(self):
        """测试配置验证成功"""
        config = Config(
            video_path=self.test_video_path,
            confidence_threshold=0.6,
            max_fps=30
        )
        
        manager = ConfigManager()
        result = manager.validate_config(config)
        assert result is True
    
    def test_validate_config_empty_video_path(self):
        """测试空视频路径验证"""
        config = Config(
            video_path="",
            confidence_threshold=0.6
        )
        
        manager = ConfigManager()
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager.validate_config(config)
        
        assert "视频文件路径不能为空" in str(exc_info.value)
    
    def test_validate_config_invalid_confidence_range(self):
        """测试置信度范围验证"""
        config = Config(
            video_path=self.test_video_path,
            confidence_threshold=2.0  # 超出范围
        )
        
        manager = ConfigManager()
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager.validate_config(config)
        
        assert "置信度阈值必须在 0.0-1.0 之间" in str(exc_info.value)
    
    def test_validate_config_invalid_fps_range(self):
        """测试帧率范围验证"""
        config = Config(
            video_path=self.test_video_path,
            confidence_threshold=0.5,
            max_fps=0  # 无效值
        )
        
        manager = ConfigManager()
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager.validate_config(config)
        
        assert "最大帧率必须是正数" in str(exc_info.value)
    
    def test_config_with_custom_config_file_path(self):
        """测试通过命令行指定配置文件路径"""
        # 创建自定义配置文件
        custom_config_path = os.path.join(self.temp_dir, "custom.yaml")
        config_data = {
            'detection': {
                'confidence_threshold': 0.9
            }
        }
        with open(custom_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        args = ["--video", self.test_video_path, "--config", custom_config_path]
        manager = ConfigManager(args=args)
        config = manager.load_config()
        
        assert config.confidence_threshold == 0.9
        assert manager.config_path == custom_config_path
    
    def test_error_handling_with_suggestions(self):
        """测试错误处理和建议信息"""
        config_dict = {
            'video_path': None
        }
        
        manager = ConfigManager()
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager._create_and_validate_config(config_dict)
        
        error = exc_info.value
        assert hasattr(error, 'message')
        assert hasattr(error, 'error_code')
        assert hasattr(error, 'suggestions')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])