# Anime Character Extractor 注：未完成品

![image](https://github.com/DivinationHW/AnimeFaceExtractor/blob/main/Yuuko.jpg)
PS：本来单纯拿来扒紫苑寺有子动画里的镜头炼丹用的，想了想完善了一下放到Github水一水。

Anime Character Extractor 是一个用于从动画视频中提取人物的Python工具。我们借鉴了各种开源库和方法的优点，通过使用OpenCV库进行人脸检测，然后从每一帧中提取人物。

## 功能特性

1. **人物提取**：可以从任何提供的视频文件中提取人物图像。
2. **多种人脸检测模式**：提供了三种不同的人脸检测模式，以适应不同的需求和情况：
   - 在每一帧上进行人脸检测
   - 计算余弦相似度，跳过相似的帧
   - 使用图像差异哈希比较，跳过相似的帧
3. **关键帧提取**：如果需要，可以只在视频的关键帧上进行处理。
4. **质量控制**：可以配置输出图像的JPEG质量。

虽然我们已经努力提高这个工具的效果，但我们相信还有许多可以改进和优化的地方。

## 快速开始

确保已经安装了所有必要的依赖包，这些包包括：`OpenCV`、`NumPy`、`PIL`、`imagehash` 和 `tqdm`。你可以通过执行以下命令来安装它们：

    pip install -r requirements.txt

然后，运行 `AnimeCharacterExtractor.py` 文件：

    python AnimeCharacterExtractor.py

按照提示输入配置参数。如果你想使用默认参数，只需按回车键即可。

程序将开始处理视频文件，并在指定的输出文件夹中生成提取出的人物图像。

关键帧提取需安装且添加ffmpeg环境。

## 参数配置

- **视频文件路径**：你想要处理的视频文件的路径。默认为 'demo.mkv'。
- **输出文件夹**：生成的人物图像将被保存在这个文件夹中。默认为 'output'。
- **级联分类器路径**：用于人脸检测的级联分类器文件的路径。默认为 'lbpcascade_animeface.xml'。
- **输出图像的JPEG质量**：生成的人物图像的JPEG质量，范围为 0-100。默认为 90。
- **人脸检测模式**：选择一个模式进行人脸检测。默认为 0。

## 致谢

我们要感谢所有为这个项目的成功做出贡献的人。特别是OpenCV、NumPy、PIL、imagehash 和 tqdm这些库的开发者，没有他们的辛勤工作和创新，这个项目就无法成为可能。我们也要特别感谢 [nagadomi](https://github.com/nagadomi/lbpcascade_animeface) 的 'lbpcascade_animeface.xml' 级联分类器项目，它对我们的工作起到了重要的推动作用。

## 许可证

此项目基于GNU General Public License v3.0许可证。

## It's the only NEET thing to do.
