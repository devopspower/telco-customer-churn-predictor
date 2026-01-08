import os
print("Current Directory:", os.getcwd())
print("Files in current directory:", os.listdir('.'))
if os.path.exists('data'):
    print("Files in 'data' folder:", os.listdir('data'))
else:
    print("The 'data' folder does not exist in this directory.")