# git_sync.py
import subprocess
import os
import sys

def run_git_command(command_list, success_message, error_message):
    """Executes a system shell command safely and returns status."""
    try:
        # shell=True is mandatory for reliable execution across Windows PowerShell environments
        result = subprocess.run(command_list, check=True, shell=True, capture_output=True, text=True)
        print(f"[SUCCESS] {success_message}")
        if result.stdout.strip():
            print(f"   👉 {result.stdout.strip().splitlines()[-1]}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[❌ ERROR] {error_message}")
        if e.stderr:
            print(f"   Details: {e.stderr.strip()}")
        return False

def main():
    print("=" * 60)
    print("🤖 PROJECT AGASTYA - REALTIME GIT MLOPS SYNC ENGINE")
    print("=" * 60)

    # 1. Verify Git status
    if not os.path.exists('.git'):
        print("[❌ CRITICAL] Directory is not an active Git repository. Exiting.")
        sys.exit(1)

    # 2. Stage all modifications
    if not run_git_command(["git", "add", "."], "All current project file variants staged successfully.", "Failed to stage file changes."):
        sys.exit(1)

    # 3. Dynamic Custom Commit Message Selection
    print("\n📝 Enter your commit message (Press ENTER to use automated fallback):")
    custom_msg = input("> ").strip()
    
    if not custom_msg:
        custom_msg = "feat(mlops): synchronized workspace layers and model asset pointers"
    
    # 4. Commit current staging cache
    if not run_git_command(["git", "commit", "-m", f'"{custom_msg}"'], f"Changes committed safely: {custom_msg}", "Commit rejected. (You might not have any new structural variations to save)."):
        print("[INFO] Synchronization cycle terminated. No workspace modifications detected.")
        sys.exit(0)

    # 5. Execute Cloud Rebase to clear 'Fetch First' errors
    print("\n📡 Fetching remote adjustments and aligning cloud state trees...")
    if not run_git_command(["git", "pull", "origin", "main", "--rebase"], "Local repository history updated and rebased cleanly with origin/main.", "Rebase conflict detected from GitHub web entries."):
        print("[HELP] If automatic rebasing fails, run 'git rebase --abort' and fetch updates manually.")
        sys.exit(1)

    # 6. Push final compiled pointer arrays
    print("\n🚀 Pushing streaming pointer layers and code to GitHub remote...")
    if run_git_command(["git", "push", "origin", "main"], "Core architectures and Git LFS binary files pushed completely live!", "Failed to push refs to remote host pipeline."):
        print("\n✨ Workspace synchronization complete! Your GitHub repository is completely up to date.")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()