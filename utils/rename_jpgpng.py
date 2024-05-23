import os


def rename_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            # 构建新的文件名
            new_name = os.path.join(folder_path, filename[:-4] + ".png")

            # 重命名文件
            os.rename(os.path.join(folder_path, filename), new_name)
            print(f"Renamed: {filename} to {new_name}")


folder_path = '/home/server/zpf/best_code_cod/Dataset/SOD/TestDatasetSOD/STERE_RGB/GT/'

rename_images(folder_path)
