import os


path = 'dataset/val/'
files = os.listdir(path)
print('files')

n = 0
for i in files:
    old_name = path+files[n]
    new_name = path+str(n+1).zfill(3)+'.jpg'
    os.rename(old_name, new_name)
    print(old_name+'>>>'+new_name)
    n = n + 1