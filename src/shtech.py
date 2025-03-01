import shutil
import kagglehub

def main():
    # Download the latest version of the dataset
    path = kagglehub.dataset_download("tthien/shanghaitech") + "/ShanghaiTech"
    
    # Define the target directory and ensure it exists
    target_dir = "./data/ShanghaiTech"
    
    # Uncompress the downloaded ZIP file into the target directory
    shutil.copytree(path, target_dir)
    
    print(f"Dataset extracted to {target_dir}")

if __name__ == '__main__':
    main()
