from pathlib import Path; import shutil
if not Path(".env").exists():
    shutil.copyfile(".env.example", ".env")
print("✅ .env created. Edit credentials before running.")
