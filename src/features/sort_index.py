import glob


def get_file_number(val):
    """
        Args:
            val (path fo bmp file)  | ../../data/Tanimoto_eLife_Fig3B_cp/2012/main/img0.bmp
        Return:
            file_number (int)       | 0
    """
    file_number = val.split("img")[1].split(".bmp")[0]
    return int(file_number)


def get_binaryfile_number(val):
    """
        Args:
            val (path fo bmp file)  | ../../data/processed/alldata/date_000000.pt
        Return:
            file_number (int)       | 0
    """
    file_number = val.split("/")[-1].split("_")[1].split(".pt")[0]
    return int(file_number)


if __name__ == "__main__":
    img_path_ls = glob.glob("../../data/processed/alldata/*")
    img_path_ls.sort(key=get_file_number)
    print(img_path_ls[:10])
