import sys
import shutil

def main ():
    file_code =sys.argv[1]
    dst_path="videos/keepers/"+ file_code+".mp4"
    src_path = "videos/" + file_code+".mp4"
    shutil.move(src_path, dst_path)