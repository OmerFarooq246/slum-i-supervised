import gdown

url = "https://drive.google.com/drive/folders/1yhDwR4zyPQO78x040uGCPqFarTDQ3yQm"
target_folder = "/netscratch/mukhtar/ncode_slum_i_taha-main/patrick_data"

gdown.download_folder(url, output=target_folder, quiet=False)
