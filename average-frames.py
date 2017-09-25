import os

image_path = os.getcwd() + '/image/'
folders = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a08', 'a09', 'a11', 'a12']
image_numbers = 0
vaild_folders = 0
for folder in folders:
    folder_path = image_path + folder + '/'
    image_folders = list(os.listdir(folder_path))
    print image_folders
    for images in image_folders:
        if os.path.exists(folder_path + images + '/'):
            num = len(list(os.walk(folder_path + images + '/'))[0][2])
            image_numbers += num
            vaild_folders += 1

average_frames = image_numbers/vaild_folders

print average_frames