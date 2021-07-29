from facenet_pytorch import MTCNN, InceptionResnetV1

def nnet():
    mtcnn0 = MTCNN(image_size= 240, margin=0,keep_all= False, min_face_size=30) 
    mtcnn1 = MTCNN(image_size=240,keep_all=True,margin=0,min_face_size=30)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    return mtcnn0,mtcnn1,resnet 