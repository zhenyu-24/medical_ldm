from omegaconf import OmegaConf, DictConfig
from typing import Any


class Config:
    def __init__(self):
        self.model: DictConfig = None  # 使用类型注解指定 model 属性的类型


def load_yaml_to_config(yaml_file_path: str) -> Config:
    """
    从给定的 YAML 文件路径加载配置到 Config 对象中。

    参数:
        yaml_file_path (str): YAML 配置文件的路径。

    返回:
        Config: 包含加载配置的 Config 对象。
    """
    cfg = Config()
    try:
        # 尝试加载 YAML 文件并将其内容赋值给 cfg.model
        cfg.model = OmegaConf.load(yaml_file_path)


    except FileNotFoundError:
        print(f"Error: 文件 {yaml_file_path} 未找到。")
    except Exception as e:
        print(f"Error: 加载配置文件时出错 - {e}")

    return cfg


# 示例使用
if __name__ == "__main__":
    yaml_file_path = 'config/rqgan_2d.yaml'
    config = load_yaml_to_config(yaml_file_path)
    if config.model is not None:
        print(config.model)