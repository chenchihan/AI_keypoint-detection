from torch.utils.data.dataset import Dataset
import PIL.Image as Image
import cv2
import os
import numpy as np
import shutil
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler
from math import dist
import glob
import csv



# 複製圖片

def objcopy(img_root, path,img_root2=None):
    if img_root2 != None:
        files= os.listdir(img_root)
        files.sort(key=lambda x:int(x.split('.')[0]))
        n=1
        for file in files:
            if len(os.path.splitext(files[0])[0])==4:
                filename=os.path.join(img_root, file)
                newname=(path+'/'+'{0:04}'.format(n)+'.png')
                n+=1
                shutil.copy(filename,newname)
            else:
                filename=os.path.join(img_root, file)
                newname=(path+'/'+str(n)+'.png')
                n+=1
                shutil.copy(filename,newname)

        files2= os.listdir(img_root2)
        files2.sort(key=lambda x:int(x.split('.')[0]))
        for file2 in files2:
            if len(os.path.splitext(files2[0])[0])==4:
                filename1=os.path.join(img_root2, file2)
                newname2=(path+'/'+'{0:04}'.format(n)+'.png')
                n+=1
                shutil.copy(filename1,newname2)
            else:
                filename1=os.path.join(img_root2, file2)
                newname2=(path+'/'+str(n)+'.png')
                n+=1
                shutil.copy(filename1,newname2)

    else :
        files= os.listdir(img_root)
        files.sort(key=lambda x:int(x.split('.')[0]))
        n=1
        for file in files:
            if len(os.path.splitext(files[0])[0])==4:
                filename=os.path.join(img_root, file)
                newname=(path+'/'+'{0:04}'.format(n)+'.png')
                n+=1
                shutil.copy(filename,newname)
            else:
                filename=os.path.join(img_root, file)
                newname=(path+'/'+str(n)+'.png')
                n+=1
                shutil.copy(filename,newname)

# 導入訓練圖片

def train_dataset(img_root, label_root,img_root2=None,label_root2=None):
    imgs = []
    if img_root2 != None:
        if not(os.path.exists('./mix_image/')) or not os.listdir('./mix_image/'):
            os.mkdir('mix_image')
            path='./mix_image/'
            os.mkdir('mix_label')
            path2='./mix_label/'
            # 複製第1、2個image
            objcopy(img_root, path,img_root2)
            # 複製第1、2個label
            objcopy(label_root, path2,label_root2)
        else :
            shutil.rmtree('mix_image')
            os.mkdir('mix_image')
            path='./mix_image/'
            # 複製第1、2個image
            objcopy(img_root, path,img_root2)
            shutil.rmtree('mix_label')
            os.mkdir('mix_label')
            path2='./mix_label/'
            # 複製第1、2個label
            objcopy(label_root, path2,label_root2)
        n2 = len(os.listdir(path))
        for i in range(n2):
            img_file=os.listdir(path)

            img_file.sort(key=lambda x:int(x.split('.')[0]))

            img = os.path.join(path,img_file[i])

            label_file=os.listdir(path2)

            label_file.sort(key=lambda x:int(x.split('.')[0]))

            label = os.path.join(path2,label_file[i])

            imgs.append((img, label))

        return imgs
    else :
        n = len(os.listdir(img_root))
        for i in range(n):
            img_file=os.listdir(img_root)

            img_file.sort(key=lambda x:int(x.split('.')[0]))

            img = os.path.join(img_root,img_file[i])

            label_file=os.listdir(label_root)

            label_file.sort(key=lambda x:int(x.split('.')[0]))

            label = os.path.join(label_root,label_file[i])

            imgs.append((img, label))

        return imgs


def test_dataset(img_root):
    imgs = []
    if os.path.isfile(img_root)==True:
        img = img_root
        imgs.append(img)
    else:
        FileNameList=glob.glob(img_root + "/*.jpg")
        if FileNameList==[]:
            FileNameList=glob.glob(img_root + "/*.png")
        FileNameList.sort()
        FileNameList.sort(key=len)
        n = len(FileNameList)
        for i in range(n):
            img = FileNameList[i]
            imgs.append(img)
    return imgs

def Validation_dataset(img_root, label_root):
    imgs = []
    if not(os.path.exists('./Validation_image/')) or not os.listdir('./Validation_image/'):
        os.mkdir('Validation_image')
        path='./Validation_image/'
        os.mkdir('Validation_label')
        path2='./Validation_label/'
        # 複製第1個image
        objcopy(img_root, path)
        # 複製第1個label
        objcopy(label_root, path2)
    else :
        shutil.rmtree('Validation_image')
        os.mkdir('Validation_image')
        path='./Validation_image/'
        # 複製第1個image
        objcopy(img_root, path)
        shutil.rmtree('Validation_label')
        os.mkdir('Validation_label')
        path2='./Validation_label/'
        # 複製第1個label
        objcopy(label_root, path2)
    n = len(os.listdir(path))
    for i in range(n):
        img_file=os.listdir(img_root)

        img_file.sort(key=lambda x:int(x.split('.')[0]))

        img = os.path.join(img_root,img_file[i])

        label_file=os.listdir(label_root)

        label_file.sort(key=lambda x:int(x.split('.')[0]))

        label = os.path.join(label_root,label_file[i])

        imgs.append((img, label))
    return imgs


class TrainDataset(Dataset):
    def __init__(self, img_root,label_root, img_root2=None,label_root2=None,
                 transform=None, target_transform=None):
        imgs = train_dataset(img_root, label_root,img_root2,label_root2)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, label_path = self.imgs[index]
        img_x = Image.open(img_path)
        img_y = Image.open(label_path).convert('L')
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

class TestDataset(Dataset):
    def __init__(self, img_root,transform=None, target_transform=None):
        imgs = test_dataset(img_root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_x = Image.open(img_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        return img_x

    def __len__(self):
        return len(self.imgs)

class Validation(Dataset):
    def __init__(self, img_root,label_root,transform=None, target_transform=None):
        imgs = Validation_dataset(img_root, label_root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, label_path = self.imgs[index]
        img_x = Image.open(img_path)
        img_y = Image.open(label_path).convert('L')
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

class CMCTrainDataset(Dataset):
    def __init__(self, img_root,mask_root,transform=None, target_transform=None):
        imgs = train_dataset(img_root, mask_root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        # load images and masks
        img_path, mask_path = self.imgs[idx]
        img = np.array(Image.open(img_path).resize((1024, 1024)))
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path).convert('L').resize((1024, 1024))
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        m=np.zeros([1024,1024])
        m[mask == obj_ids[0]]=1
        m[mask == obj_ids[1]]=2
        m[mask == obj_ids[2]]=3

        # convert everything into a torch.Tensor
        masks = torch.as_tensor(m, dtype=torch.uint8)
        if self.transform is not None:
            img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(masks)

        return img, masks

    def __len__(self):
        return len(self.imgs)



class KeypointTrainDataset(Dataset):
    def __init__(self, img_root,mask_root,transform=None, target_transform=None):
        imgs = train_dataset(img_root, mask_root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        # load images and masks
        img_path, mask_path = self.imgs[idx]
        img = np.array(Image.open(img_path).resize((1024, 1024)))
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path).convert('L').resize((1024, 1024))
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        m=np.zeros([1024,1024])
        m[mask == obj_ids[0]]=1
        m[mask == obj_ids[1]]=2
        m[mask == obj_ids[2]]=3
        m[mask == obj_ids[3]]=4
        m[mask == obj_ids[4]]=5

        # convert everything into a torch.Tensor
        masks = torch.as_tensor(m, dtype=torch.uint8)
        if self.transform is not None:
            img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(masks)

        return img, masks

    def __len__(self):
        return len(self.imgs)










# 將資料分成訓練跟測試

def dataset_sampler(dataset, val_percentage=0.1):
    """
    split dataset into train set and val set
    :param dataset:
    :param val_percentage: validation percentage
    :return: split sampler
    """
    sample_num = len(dataset)
    file_idx = list(range(sample_num))
    train_idx, val_idx = train_test_split(file_idx, test_size=val_percentage, random_state=42)
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    return train_sampler, val_sampler


def load_split_train_test(datadir,valid_size = 0.1):

    num_train = len(datadir)                               # 训练集数量
    indices = list(range(num_train))                          # 训练集索引

    split = int(np.floor(valid_size * num_train))             # 获取20%数据作为验证集
    np.random.shuffle(indices)                                # 打乱数据集

    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]    # 获取训练集，测试集
    train_sampler = SubsetRandomSampler(train_idx)            # 打乱训练集，测试集
    val_sampler  = SubsetRandomSampler(test_idx)
    return train_sampler, val_sampler


# 將資料分成訓練、測試和驗證集

def train_test_val_split(dataset, ratio_train=0.7, ratio_test=0.1, ratio_val=0.2):
    train, middle = train_test_split(dataset, train_size=ratio_train, test_size=ratio_test + ratio_val)
    ratio = ratio_val/(1-ratio_train)
    test, validation = train_test_split(middle, test_size=ratio)
    return train, test, validation



def imageadd(img1,img2,color=False):
    img = Image.open(img1).convert('RGB').resize((1024, 1024))
    mask = Image.open(img2).convert('RGB').resize((1024, 1024))
    if color==True:
        x,y = mask.size
        pim = mask.load()
        for w in range(x):
            for h in range(y):
                r,g,b = pim[w,h]
                if r > 200:
                    mask.putpixel((w, h),(0,0 ,255))
                elif r > 100 and r<200:
                    mask.putpixel((w, h), (0,255,0))
                elif r> 50 and r<100:
                    mask.putpixel((w, h), (255,0,0))
    img_compose=Image.blend(img,mask,alpha=0.5)
    return img_compose



#  json to csv

def j2csv(data,target='shapes'):

    d = json.load(open(data))

    df = pd.DataFrame(d[target])
    file_path = os.path.splitext(data)[0]
    df.to_csv(file_path+'.csv', encoding='utf-8', index=False)



# 更改json 圖片路徑
#定义操作函数
def change_json(path):
    files=os.listdir(path)
    files.sort(key=lambda x:int(x.split('.')[0]))
    #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
    for file in files:
        dir=os.path.join(path,file)
        # os.path.join()，将join()里面得参数拼接成一个完整得路径。
        # 检查是否为文件夹，如果是，则递归
        if os.path.isdir(dir):
            chang_json(dir)
            continue
        file_split=file.split('.')
        #file.split将file列表数据以"."分割，并赋值给file_split
        if file_split[-1] == "json":
            str=path+"\\"+"".join(file_split[0])+".jpg" # 定义要更改的文件名
            with open(path+'\\'+file,'rb') as load_f:
            #定义为只读模式，并定义名称为f
                params = json.load(load_f)
                #加载json文件中的内容给params
                load_f.close() # 关闭文件
            with open(path+'\\'+file,'w') as dump_f:
            #定义为写入模式，并定义名称为f
                print(str) # 查看要写入的名称
                params['imagePath'] = str # 更改参数
                json.dump(params,dump_f) # 将params写入文件
                dump_f.close() #关闭文件

def get_coco_paths(coco_root, dataset):
    # Read the json file
    with open(os.path.join(coco_root, f'annotations/captions_{dataset}.json'), 'r') as f:
        annotations = json.load(f)

    # Store captions and image names in vectors
    coco_image_paths = []
    for annot in tqdm.tqdm(annotations['annotations']):
        image_id = annot['image_id']
        full_coco_image_path = os.path.join(coco_root, dataset, f'{image_id:012d}.jpg')
        coco_image_paths.append(full_coco_image_path)

    return coco_image_paths



# def data_save(num,img_id,old_keypoints,path,keypoints=None,mood="csv"):
#     if mood=="csv":
#         if keypoints!=None:
#             with open(path+'/'+str(num)+'.csv', 'w', encoding='UTF8', newline='') as f:
#                 writer = csv.writer(f)
#                 dis_list=[]
#                 for i in range(len(old_keypoints[0])):
#                     dis=dist(old_keypoints[0][i],keypoints[0][i])
#                     dis=round(dis, 2)
#                     dis_list.append(dis)
#                 distan1=distanline(old_keypoints[0][0],old_keypoints[0][1],old_keypoints[0][2],old_keypoints[0][3])
#                 distan2=distanline(keypoints[0][0],keypoints[0][1],keypoints[0][2],keypoints[0][3])
#                 datas=[['image-ID'],[str(img_id)],['label_keypoint'],[old_keypoints[0]],
#                 ['predict_keypoint'],[keypoints[0]],
#                 ["distance"],[dis_list],["original_subluxation(%)"],[distan1],["predict_subluxation(%)"],[distan2]]
#                 for data in datas:
#                     writer.writerow(data)
#         else:
#             with open(path+'/'+str(num)+'.csv', 'w', encoding='UTF8', newline='') as f:
#                 writer = csv.writer(f)
#                 distan1=distanline(old_keypoints[0][0],old_keypoints[0][1],old_keypoints[0][2],old_keypoints[0][3])
#                 datas=[['image-ID'],[str(img_id)],['predict_keypoint'],[old_keypoints[0]],
#                 ["predict_subluxation(%)"],[distan1]]
#                 for data in datas:
#                     writer.writerow(data)
#     else:
#         if keypoints!=None:
#             file= open(path+'/'+str(num)+'.txt','w')
#             dis_list=[]
#             for i in range(len(old_keypoints[0])):
#                 dis=dist(old_keypoints[0][i],keypoints[0][i])
#                 dis=round(dis, 2)
#                 dis_list.append(dis)
#             distan1=distanline(old_keypoints[0][0],old_keypoints[0][1],old_keypoints[0][2],old_keypoints[0][3])
#             distan2=distanline(keypoints[0][0],keypoints[0][1],keypoints[0][2],keypoints[0][3])
#             datas=['image-ID',str(img_id),'label_keypoint',old_keypoints[0],
#             'predict_keypoint',keypoints[0],
#             "distance",dis_list,"original_subluxation(%)",distan1,"predict_subluxation(%)",distan2]
#             for data in datas:
#                 file.write(str(data))
#                 file.write('\n')
#             file.close()
#         else:
#             file= open(path+'/'+str(num)+'.txt','w')
#             distan1=distanline(old_keypoints[0][0],old_keypoints[0][1],old_keypoints[0][2],old_keypoints[0][3])
#             datas=['image-ID',str(img_id),'predict_keypoint',old_keypoints[0],"predict_subluxation(%)",distan1]
#             for data in datas:
#                 file.write(str(data))
#                 file.write('\n')
#             file.close()
def data_save(num,img_id,old_keypoints,path,keypoints=None,mood="csv"):
    if mood=="csv":
        if keypoints!=None:
            with open(path+'/'+str(num)+'.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                dis_list=[]
                for i in range(len(old_keypoints[0])):
                    dis=dist(old_keypoints[0][i],keypoints[0][i])
                    dis=round(dis, 2)
                    dis_list.append(dis)
                if len(old_keypoints[0])==5:
                    distan1=distanline(old_keypoints[0][0],old_keypoints[0][1],old_keypoints[0][2],old_keypoints[0][3])
                    distan2=distanline(keypoints[0][0],keypoints[0][1],keypoints[0][2],keypoints[0][3])
                    datas=['image-ID',str(img_id),'label_keypoint',old_keypoints[0],
                           'predict_keypoint',keypoints[0],
                           "distance",dis_list,"original_subluxation(%)",distan1,"predict_subluxation(%)",distan2]
                else:
                    distan1=distanline(old_keypoints[0][0],old_keypoints[0][1],old_keypoints[0][2],old_keypoints[0][3])
                    distan2=distanline(keypoints[0][0],keypoints[0][1],keypoints[0][2],keypoints[0][3])
                    distan3=distanline(old_keypoints[0][5],old_keypoints[0][6],old_keypoints[0][7],old_keypoints[0][8])
                    distan4=distanline(keypoints[0][5],keypoints[0][6],keypoints[0][7],keypoints[0][8])
                    datas=['image-ID',str(img_id),'label_keypoint',old_keypoints[0],
                    'predict_keypoint',keypoints[0],
                    "distance",dis_list,"right_original_subluxation(%)",distan1,"right_predict_subluxation(%)",distan2,
                    "left_original_subluxation(%)",distan3,"left_predict_subluxation(%)",distan4]
                for data in datas:
                    writer.writerow(data)
            f.close()
        else:
            with open(path+'/'+str(num)+'.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                if len(old_keypoints[0])==5:
                    distan1=distanline(old_keypoints[0][0],old_keypoints[0][1],old_keypoints[0][2],old_keypoints[0][3])
                    datas=['image-ID',str(img_id),'predict_keypoint',old_keypoints[0],
                    "predict_subluxation(%)",distan1]
                else:
                    distan1=distanline(old_keypoints[0][0],old_keypoints[0][1],old_keypoints[0][2],old_keypoints[0][3])
                    distan2=distanline(old_keypoints[0][5],old_keypoints[0][6],old_keypoints[0][7],old_keypoints[0][8])
                    datas=['image-ID',str(img_id),'predict_keypoint',old_keypoints[0],
                           "right_predict_subluxation(%)",distan1,
                           "left_predict_subluxation(%)",distan2]
                for data in datas:
                    writer.writerow(data)
            f.close()
    else:
        if keypoints!=None:
            file= open(path+'/'+str(num)+'.txt','w')
            dis_list=[]
            for i in range(len(old_keypoints[0])):
                dis=dist(old_keypoints[0][i],keypoints[0][i])
                dis=round(dis, 2)
                dis_list.append(dis)
            if len(old_keypoints[0])==5:
                distan1=distanline(old_keypoints[0][0],old_keypoints[0][1],old_keypoints[0][2],old_keypoints[0][3])
                distan2=distanline(keypoints[0][0],keypoints[0][1],keypoints[0][2],keypoints[0][3])
                datas=['image-ID',str(img_id),'label_keypoint',old_keypoints[0],
                        'predict_keypoint',keypoints[0],
                        "distance",dis_list,"original_subluxation(%)",distan1,"predict_subluxation(%)",distan2]
            else:
                distan1=distanline(old_keypoints[0][0],old_keypoints[0][1],old_keypoints[0][2],old_keypoints[0][3])
                distan2=distanline(keypoints[0][0],keypoints[0][1],keypoints[0][2],keypoints[0][3])
                distan3=distanline(old_keypoints[0][5],old_keypoints[0][6],old_keypoints[0][7],old_keypoints[0][8])
                distan4=distanline(keypoints[0][5],keypoints[0][6],keypoints[0][7],keypoints[0][8])
                datas=['image-ID',str(img_id),'label_keypoint',old_keypoints[0],
                       'predict_keypoint',keypoints[0],
                       "distance",dis_list,"right_original_subluxation(%)",distan1,"right_predict_subluxation(%)",distan2,
                       "left_original_subluxation(%)",distan3,"left_predict_subluxation(%)",distan4]
            for data in datas:
                file.write(str(data))
                file.write('\n')
            file.close()
        else:
            file= open(path+'/'+str(num)+'.txt','w')
            if len(old_keypoints[0])==5:
                distan1=distanline(old_keypoints[0][0],old_keypoints[0][1],old_keypoints[0][2],old_keypoints[0][3])
                datas=['image-ID',str(img_id),'predict_keypoint',old_keypoints[0],
                       "predict_subluxation(%)",distan1]
            else:
                distan1=distanline(old_keypoints[0][0],old_keypoints[0][1],old_keypoints[0][2],old_keypoints[0][3])
                distan2=distanline(old_keypoints[0][5],old_keypoints[0][6],old_keypoints[0][7],old_keypoints[0][8])
                datas=['image-ID',str(img_id),'predict_keypoint',old_keypoints[0],
                        "right_predict_subluxation(%)",distan1,
                        "left_predict_subluxation(%)",distan2]
            for data in datas:
                file.write(str(data))
                file.write('\n')
            file.close()

def getLinearEquation(a,b):
    p1x=a[0]
    p1y=a[1]
    p2x=b[0]
    p2y=b[1]
    sign = 1
    a = p2y - p1y
    if a < 0:
        sign = -1
        a = sign * a
    b = sign * (p1x - p2x)
    c = sign * (p1y * p2x - p1x * p2y)
    return [a, b, c]

def point_distance_line(point,line_point1,line_point2):
	#计算向量
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1,vec2)) / np.linalg.norm(line_point1-line_point2)
    return distance

def distanline(a,b,c,d):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    d=np.array(d)
    da=point_distance_line(c,a,b)
    db=point_distance_line(d,a,b)
    dc=da/db*100
    if db ==0:
        dc = 0.0
    dc=round(dc, 2)
    return dc





def visualize(image,keypoints,path,keypoints2=None,num=None,name=None,color=(0,0,255)):
    fontsize = 10
    keypoints_classes_ids2names = {0: 'metacarpal-L', 1: 'metacarpal-R',2:"trapezium",3:"metacarpal-BL",4:"finger"}
    if keypoints2 !=None:
        for kps in keypoints:
            for idx, kp in enumerate(kps):
                image = cv2.circle(image.copy(), tuple(kp), 1,(255,0,0),1)
                # image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)
            for kps2 in keypoints2:
                for idx, kp2 in enumerate(kps2):
                    image = cv2.circle(image.copy(), tuple(kp2), 1,color,1)
        filename=path+"/"+name+str(num)+'.png'
        cv2.imwrite(filename, image)
    else:
        for kps in keypoints:
            for idx, kp in enumerate(kps):
                image = cv2.circle(image.copy(), tuple(kp), 1,(255,255,255), 1)
                # image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)
        # filename=path+"/"+name+str(num)+'.png'
        filename=path+"/"+name+str(num)+'.png'
        cv2.imwrite(filename, image)

    # if image_original is None and keypoints_original is None:
    #     plt.figure(figsize=(40,40))
    #     plt.imshow(image)

    # else:
    #     for bbox in bboxes_original:
    #         start_point = (bbox[0], bbox[1])
    #         end_point = (bbox[2], bbox[3])
    #         image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0,255,0), 2)
        
    #     for kps in keypoints_original:
    #         for idx, kp in enumerate(kps):
    #             image_original = cv2.circle(image_original, tuple(kp), 5, (255,0,0), 10)
    #             image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)

    #     f, ax = plt.subplots(1, 2, figsize=(40, 20))

    #     ax[0].imshow(image_original)
    #     ax[0].set_title('Original image', fontsize=fontsize)

    #     ax[1].imshow(image)
    #     ax[1].set_title('Transformed image', fontsize=fontsize)


def valuemean(path, n=7):
    p1=[]
    p2=[]
    p3=[]
    p4=[]
    p5=[]
    mean=[]
    data_path = os.path.join(path)
    data_list_path = glob.glob(data_path + "/*.txt")
    num=len(data_list_path)
    for data in data_list_path:
        value=readFile(data,num=n,mood=0)
        pt1,pt2,pt3,pt4,pt5=value[0],value[1],value[2],value[3],value[4]
        m=sum(value)/len(value)
        p1.append(pt1)
        p2.append(pt2)
        p3.append(pt3)
        p4.append(pt4)
        p5.append(pt5)
        mean.append(m)
    p1_mean=round(sum(p1)/len(p1),2)
    p2_mean=round(sum(p2)/len(p2),2)
    p3_mean=round(sum(p3)/len(p3),2)
    p4_mean=round(sum(p4)/len(p4),2)
    p5_mean=round(sum(p5)/len(p5),2)
    mean_mean=round(sum(mean)/len(mean),2)
    file= open(path+'/'+'mean_loss.txt','w')
    datas=['metacarpal-L',p1_mean,'metacarpal-R',p2_mean,
           'trapezium',p3_mean,'metacarpal-BL',p4_mean,
           "index",p5_mean,'per_mean',mean_mean,
           'total_num',num]
    for data in datas:
        file.write(str(data))
        file.write('\n')
    file.close()
    return p1,p2,p3,p4,p5,mean,num



def readFile(fileName,num,mood=1):
        fileObj = open(fileName, "r") #opens the file in read mode
        words = fileObj.read().splitlines() #puts the file into an array
        if mood==1:
            a=words[num][1:-1].split(',')
            word=[float(aa) for aa in a]
        elif mood==2:
            a=words[num][0:].split(',')
            word=[float(aa) for aa in a]
        else:
            a=words[num][1:-1].split(',')
            lis=[]
            ch="[ ]"
            for i in range(len(a)):
                string = ''.join( x for x in a[i] if x not in ch)
                lis.append(string)
            word=[float(aa) for aa in lis]
        fileObj.close()
        return word