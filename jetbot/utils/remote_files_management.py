import os
import paramiko
import stat

# --- CONFIGURATION ---
REMOTE_HOST = "192.168.55.1"  # Replace with your remote machine's local IP
REMOTE_PORT = 22  # Standard SSH/SFTP port
# REMOTE_PORT = 2222  # Standard SSH/SFTP port
REMOTE_USER = "cuterbot"  # Remote SSH username
REMOTE_PASSWORD = "cuterbot"  # Remote SSH password (or use private_key_path)

REMOTE_DIR = "/home/cuterbot/Cuterbot/notebooks/road_following/dataset_xy/"  # Source directory on remote host
REMOTE_DATA_FILE = "road_following_dataset_xy_2024-12-25_04-45-04.zip"
LOCAL_DIR = "./local_download_folder"  # Destination directory on local host

REMOTE_DIR_REPO = "/home/cuterbot/model_repo"
REMOTE_DIR_REPO_RC = os.path.join(REMOTE_DIR_REPO, "road_following")
LOCAL_DIR_REPO = 'D:\\AI_Lecture_Demos\\Data_Repo\\GPU\\models_repo'

# ---------------------

def download_files_from_remote(remote_host=REMOTE_HOST, remote_dir=REMOTE_DIR, remote_data_file=REMOTE_DATA_FILE,  local_path=LOCAL_DIR):
    # Ensure local directory exists
    if not os.path.exists(local_path):
        os.makedirs(local_path)
        print(f"Created local directory for keep the training data set: {LOCAL_DIR}")

    transport = None
    sftp = None
    try:
        print(f"Connecting to Nano on {remote_host}...")
        # Initialize SSH transport session
        transport = paramiko.Transport((remote_host, REMOTE_PORT))
        transport.connect(username=REMOTE_USER, password=REMOTE_PASSWORD)

        # Initialize SFTP client
        sftp = paramiko.SFTPClient.from_transport(transport)

        print(f"Changing remote directory to: {remote_dir}")
        sftp.chdir(remote_dir)

        remote_file_path = os.path.join(remote_dir, remote_data_file).replace("\\", "/")
        local_file_path = os.path.join(local_path, remote_data_file)
        file_stat = sftp.stat(remote_file_path)
        if not stat.S_ISDIR(file_stat.st_mode):
            print(f"Downloading: {remote_file_path} ...")
            sftp.get(remote_file_path, local_file_path)
        else:
            print(f"Can not download the dataset! Check {remote_file_path} is a dataset file path or is existing on the remote host.?")

        '''
        # List files in the remote directory
        files = sftp.listdir()
        print(f"Found {len(files)} items in remote directory.")
        
        for file_name in files:
            remote_file_path = os.path.join(remote_dir, file_name).replace("\\", "/")
            local_file_path = os.path.join(local_path, file_name)

            # Check if the item is a file (skips subdirectories)
            file_stat = sftp.stat(remote_file_path)
            if not stat.S_ISDIR(file_stat.st_mode):
                print(f"Downloading: {file_name} ...")
                sftp.get(remote_file_path, local_file_path)
            else:
                print(f"Skipping directory: {file_name}")
        '''
        print("🎉 The data set files is loaded from Nano successfully!")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

    finally:
        # Securely close connections
        if sftp:
            sftp.close()
        if transport:
            transport.close()
        print("Connection to Nano closed.")

def save_files_to_remote(remote_host=REMOTE_HOST, remote_dir=REMOTE_DIR_REPO, local_dir=LOCAL_DIR_REPO):
    transport = None
    sftp = None
    try:
        print(f"Connecting to Nano on {remote_host}...")
        # Initialize SSH transport session
        transport = paramiko.Transport((remote_host, REMOTE_PORT))
        transport.connect(username=REMOTE_USER, password=REMOTE_PASSWORD)
        # Initialize SFTP client
        sftp = paramiko.SFTPClient.from_transport(transport)
        remote_dir_rc = os.path.join(remote_dir, "road_following").replace("\\", "/")
        remote_dir_od = os.path.join(remote_dir, "object_detection").replace("\\", "/")
        try:
            # Check if the directory already exists
            sftp.stat(remote_dir)
            print(f"Directory '{remote_dir}' already exists.")
        except IOError:
            # If stat() throws an IOError, the directory does not exist, so we create it
            sftp.mkdir(remote_dir)
            print(f"Directory '{remote_dir}' created successfully.")

        try:
            # Check if the directory already exists
            sftp.stat(remote_dir_rc)
            print(f"Directory '{remote_dir_rc}' already exists.")
        except IOError:
            # If stat() throws an IOError, the directory does not exist, so we create it
            sftp.mkdir(remote_dir_rc)
            print(f"Directory '{remote_dir_rc}' created successfully.")

        try:
            # Check if the directory already exists
            sftp.stat(remote_dir_od)
            print(f"Directory '{remote_dir_od}' already exists.")
        except IOError:
            # If stat() throws an IOError, the directory does not exist, so we create it
            sftp.mkdir(remote_dir_od)
            print(f"Directory '{remote_dir_od}' created successfully.")

        upload_dir_recursive(sftp, local_dir, remote_dir)
        print("🚀 Directory sync complete!")

    except Exception as err:
        print(f"Failed to connect to Nano on {remote_host}, with error: {err}")

    finally:
        # Securely close connections
        if sftp:
            sftp.close()
        if transport:
            transport.close()

def upload_dir_recursive(sftp, local_dir, remote_dir):
    for item in os.listdir(local_dir):
        local_path = os.path.join(local_dir, item)
        # Always use forward slashes for remote Linux paths
        remote_path = f"{remote_dir}/{item}"

        if os.path.isdir(local_path):
            # If it's a folder, recurse into it
            upload_dir_recursive(sftp, local_path, remote_path)
        else:
            # If it's a file, upload it
            try:
                sftp.put(local_path, remote_path)
                print(f"Uploaded: {local_path} -> {remote_path}")
            except Exception as e:
                print(f"Failed to upload {local_path}: {e}")
