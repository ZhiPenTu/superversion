"""
自定义异常类定义

定义系统中使用的各种异常类，提供详细的错误信息和错误处理机制。
"""


class VideoObjectDetectionError(Exception):
    """视频物体检测系统基础异常类
    
    所有系统异常的基类，提供统一的错误处理接口。
    """
    
    def __init__(self, message: str, error_code: str = None, suggestions: str = None):
        """初始化异常
        
        Args:
            message: 错误信息
            error_code: 错误代码
            suggestions: 错误恢复建议
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.suggestions = suggestions
    
    def __str__(self) -> str:
        """返回格式化的错误信息"""
        error_msg = f"错误: {self.message}"
        if self.error_code:
            error_msg += f" (错误代码: {self.error_code})"
        if self.suggestions:
            error_msg += f"\n建议: {self.suggestions}"
        return error_msg


class VideoProcessingError(VideoObjectDetectionError):
    """视频处理相关错误
    
    当视频文件读取、写入或处理过程中发生错误时抛出。
    """
    
    def __init__(self, message: str, error_code: str = None, suggestions: str = None):
        super().__init__(message, error_code, suggestions)


class FileNotFoundError(VideoProcessingError):
    """文件不存在错误"""
    
    def __init__(self, file_path: str):
        message = f"视频文件不存在: {file_path}"
        suggestions = "请检查文件路径是否正确，确保文件存在且有读取权限"
        super().__init__(message, "FILE_NOT_FOUND", suggestions)


class UnsupportedFormatError(VideoProcessingError):
    """不支持的文件格式错误"""
    
    def __init__(self, file_path: str, format_info: str = None):
        message = f"不支持的视频格式: {file_path}"
        if format_info:
            message += f" ({format_info})"
        suggestions = "请使用支持的视频格式，如 .mp4 文件"
        super().__init__(message, "UNSUPPORTED_FORMAT", suggestions)


class CorruptedFileError(VideoProcessingError):
    """文件损坏错误"""
    
    def __init__(self, file_path: str):
        message = f"视频文件损坏或无法读取: {file_path}"
        suggestions = "请检查文件是否完整，尝试使用其他视频播放器验证文件是否正常"
        super().__init__(message, "CORRUPTED_FILE", suggestions)


class ModelLoadError(VideoObjectDetectionError):
    """模型加载错误
    
    当物体检测模型加载失败时抛出。
    """
    
    def __init__(self, model_path: str = None, details: str = None):
        if model_path:
            message = f"无法加载检测模型: {model_path}"
        else:
            message = "无法加载默认检测模型"
        
        if details:
            message += f" - {details}"
            
        suggestions = "请检查模型文件是否存在，或尝试重新下载模型文件"
        super().__init__(message, "MODEL_LOAD_FAILED", suggestions)


class ConfigurationError(VideoObjectDetectionError):
    """配置错误
    
    当配置文件解析或配置参数验证失败时抛出。
    """
    
    def __init__(self, config_issue: str, config_path: str = None):
        if config_path:
            message = f"配置错误 ({config_path}): {config_issue}"
        else:
            message = f"配置错误: {config_issue}"
            
        suggestions = "请检查配置文件格式是否正确，或使用默认配置"
        super().__init__(message, "CONFIGURATION_ERROR", suggestions)


class DetectionError(VideoObjectDetectionError):
    """检测错误
    
    当物体检测过程中发生错误时抛出。
    """
    
    def __init__(self, frame_number: int = None, details: str = None):
        if frame_number is not None:
            message = f"第 {frame_number} 帧检测失败"
        else:
            message = "物体检测失败"
            
        if details:
            message += f": {details}"
            
        suggestions = "请检查输入帧是否有效，或尝试调整检测参数"
        super().__init__(message, "DETECTION_FAILED", suggestions)


class VisualizationError(VideoObjectDetectionError):
    """可视化错误
    
    当视频标注或可视化过程中发生错误时抛出。
    """
    
    def __init__(self, details: str = None):
        message = "视频标注失败"
        if details:
            message += f": {details}"
            
        suggestions = "请检查检测结果是否有效，或尝试重新处理"
        super().__init__(message, "VISUALIZATION_FAILED", suggestions)