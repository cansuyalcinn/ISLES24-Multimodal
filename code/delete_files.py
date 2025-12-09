import os
import subprocess

def delete_non_ncct_files_with_sudo(root_path):
    """
    Recursively traverse all patient folders and keep only files with 'ncct' in their names.
    Uses sudo for deletion if needed.
    """
    deleted_files = []
    kept_files = []
    
    # Walk through all directories and files
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        # Process files in current directory
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            
            # Keep files with 'ncct' in the name, delete others
            if 'ncct' in filename.lower():
                kept_files.append(filepath)
                print(f"[KEEP] {filepath}")
            else:
                try:
                    # Try regular delete first
                    os.remove(filepath)
                    deleted_files.append(filepath)
                    print(f"[DELETE] {filepath}")
                except PermissionError:
                    # If permission denied, use sudo
                    try:
                        subprocess.run(['sudo', 'rm', filepath], check=True)
                        deleted_files.append(filepath)
                        print(f"[DELETE with sudo] {filepath}")
                    except subprocess.CalledProcessError as e:
                        print(f"[ERROR] Could not delete {filepath}: {e}")
                except Exception as e:
                    print(f"[ERROR] Could not delete {filepath}: {e}")
        
        # Remove empty directories after processing files
        for dirname in dirnames:
            dirpath_full = os.path.join(dirpath, dirname)
            try:
                # Check if directory is empty
                if not os.listdir(dirpath_full):
                    try:
                        os.rmdir(dirpath_full)
                        print(f"[REMOVE EMPTY DIR] {dirpath_full}")
                    except PermissionError:
                        subprocess.run(['sudo', 'rmdir', dirpath_full], check=True)
                        print(f"[REMOVE EMPTY DIR with sudo] {dirpath_full}")
            except Exception as e:
                print(f"[ERROR] Could not remove directory {dirpath_full}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Files kept: {len(kept_files)}")
    print(f"  Files deleted: {len(deleted_files)}")
    print(f"{'='*60}")
    
    return kept_files, deleted_files


# Run the cleanup
root_path = '/media/cansu/DiskSpace/Cansu/ISLES24/train/raw_data'

print(f"Starting cleanup in: {root_path}")
print(f"This will DELETE all files that don't have 'ncct' in their names.")
print(f"{'='*60}\n")

# DRY RUN first
print("DRY RUN - showing what would be deleted:\n")
for dirpath, dirnames, filenames in os.walk(root_path):
    for filename in filenames:
        filepath = os.path.join(dirpath, filename)
        if 'ncct' in filename.lower():
            print(f"[WOULD KEEP] {filepath}")
        else:
            print(f"[WOULD DELETE] {filepath}")

# Uncomment to actually run:
kept, deleted = delete_non_ncct_files_with_sudo(root_path)