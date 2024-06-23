import requests
import os

# 基础URL
base_url = "https://vision.middlebury.edu/stereo/data/scenes2021/data/"

def download_file(url, save_path):
    """下载文件并保存到指定路径"""
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"文件已下载并保存到 {save_path}")
    else:
        print(f"无法下载文件：{url}")

def download_dataset(dataset_name):
    """根据数据集名称下载相关文件"""
    dataset_url = f"{base_url}{dataset_name}/"
    files_to_download = ['calib.txt', 'im0.png', 'im1.png']
    
    # 创建保存文件的目录
    save_dir = os.path.join("dataset", dataset_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
        for file_name in files_to_download:
            file_url = f"{dataset_url}{file_name}"
            save_path = os.path.join(save_dir, file_name)
            download_file(file_url, save_path)
    else:
        print(f"数据集 {dataset_name} 已经被下载过，不需要再次下载。")

    # 解析calib.txt并打印结果
    calib_path = os.path.join(save_dir, 'calib.txt')
    if os.path.exists(calib_path):
        with open(calib_path, 'r') as calib_file:
            calib_content = calib_file.read()
        print(f"以下是 {dataset_name} 数据集的相机参数和其他信息：\n{calib_content}")
    else:
        print(f"未找到 {dataset_name} 数据集的 calib.txt 文件。")
    # 返回保存文件的目录路径
    return save_dir


if __name__ == '__main__':
    dataset_name = "chess1"  # 示例数据集名称
    download_dataset(dataset_name)