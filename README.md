# Anime Character Extractor

![image](https://github.com/DivinationHW/AnimeFaceExtractor/blob/main/Yuuko.jpg)
PS：本工具最初是为了从紫苑寺有子动画中提取镜头训练Lora而创建的。后来想到可以完善并分享给大家（水一下）。

Anime Character Extractor 是一个用于从动画视频中提取人物帧的Python工具。通过使用OpenCV库进行人脸检测，然后从中提取人物帧，也有减少重复帧加速处理的功能。

## 功能特性

1. **使用差异哈希减少重复帧**：通过计算帧间差异，跳过相似的帧。。
2. **人脸检测（可选）**：可以选择是否启用人脸检测功能。

虽然已经努力提高工具的效果，但仍有许多改进和优化的空间。

## 快速开始

确保已经安装了所有必要的依赖包，这些包包括：`OpenCV`、`PIL`、`imagehash` 和 `tqdm`。你可以通过执行以下命令来安装它们：

    pip install -r requirements.txt

然后，运行 `AnimeCharacterExtractor.py` 文件：

    python AnimeCharacterExtractor.py

按照提示输入配置参数。如果你想使用默认参数，只需按回车键即可。

程序将开始处理视频文件，并在指定的输出文件夹中生成提取出的人物图像。

## 参数配置

- **动漫视频文件或文件夹路径**：输入动漫视频的文件路径或包含多个视频文件的文件夹路径。
- **调整帧差别大小的阈值**：设置帧差别的阈值，用于决定是否删除重复帧。
- **输出帧序列的文件夹名称**：设置输出帧序列的文件夹名称。
- **是否启用人脸检测**：可以选择是否启用人脸检测功能。
- **人脸识别红框的粗细**：设置人脸识别时红框的粗细。
- **输出人脸的文件夹名称**：设置输出人脸检测结果的文件夹名称。

## 致谢

我们要感谢所有为这个项目的成功做出贡献的人。特别是OpenCV、PIL、imagehash 和 tqdm这些库的开发者，没有他们的辛勤工作和创新，这个项目就无法成为可能。我们也要特别感谢 [nagadomi](https://github.com/nagadomi/lbpcascade_animeface) 的 'lbpcascade_animeface.xml' 级联分类器项目，它对我们的工作起到了重要的推动作用。

## 许可证

此项目基于GNU General Public License v3.0许可证。

## It's the only NEET thing to do.
