## 主机资源

- 建议使用一台新的主机，若使用已有主机，需备份主机系统镜像。
- 推荐使用GPU主机安装OCR服务，而不是使用CPU主机。

## 主机配置

- CPU：16核
- GPU：显存不低于8G
- 内存：不低于16G
- 磁盘：可用空间不低于80G

## 主机依赖

- openEuler 22
  ```shell
  dnf install -y mesa-libGL
  ```
- Debian/Ubuntu
  ```shell
  apt-get install -y libgl1
  ```

## 配置

  ```shell
  vim model_config/config.yaml
  ```

  ```
  # 针对rule型文档，是否多空一行
  isMoreLine: True
  # weights文件夹的上一层
  OcrWeights: ""
  DeviceMode: "gpu" # gpu
  ```

## 端口

start.sh

  ```
  -b 0.0.0.0:8001
  ```

## 启动方式

  ```shell
  ./start.sh
  ```

## 服务接口

- POST http://ip:port/ocr/task

## 开发人员安装Python 环境：

默认为CPU环境，切换 requirements.txt中 paddlepaddle 一行的注释修改为GPU。

  ```shell
  pip install -r requirements.txt
  pip install -r requirements1.txt
  
  ```
## 脚本说明
 - 下载离线系统包  dev/download_mesa_libgl.sh
 - 安装离线系统包  dev/install.sh

