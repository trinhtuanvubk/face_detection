from os import name
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
from model import nnet


def process_data(photos_path,mtcnn0,resnet) :
    dataset = datasets.ImageFolder(photos_path) 
    # accessing names of peoples from folder names
    idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} 

    def collate_fn(x):
        return x[0]

    loader = DataLoader(dataset, collate_fn=collate_fn)
    # list of names corrospoing to cropped photos
    name_list = [] 
    # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet
    embedding_list = [] 

    for img, idx in loader:
        face, prob = mtcnn0(img, return_prob=True) 
        if face is not None and prob>0.90:
            emb = resnet(face.unsqueeze(0)) 
            embedding_list.append(emb.detach()) 
            name_list.append(idx_to_class[idx])        
    # save data
    data = [embedding_list, name_list] 
    torch.save(data, 'data_embedding.pt') 

def load_data(data_path = 'data_embedding.pt'):
    load_data = torch.load(data_path)
    embedding_list = load_data[0] 
    name_list = load_data[1] 
    return embedding_list,name_list

if __name__=='__main__':
    mtcnn0, mtcnn1, resnet = nnet()
    process_data(photos_path='photos_data', mtcnn0=mtcnn0, resnet=resnet)
