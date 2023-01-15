import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch

class PatchGenerator:

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, img):
        num_channels = img.size(0) # 3x256x256
        # unfold이용하면, 세로로 한번 자르고, 가로로 한번 잘라진다. -> 정육면체 여러개 (3x16x16x16x16) -> reshape으로 3x256x16x16으로 만들어주기
        # x.unfold(dimension, size of each slice, stride)
        patches = img.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size).reshape(num_channels, -1, self.patch_size, self.patch_size)
        # 개수를 앞쪽으로 빼주기 위해서 permute() 이용 -> 256x3x16x16(3x16x16이 256개)
        patches = patches.permute(1,0,2,3)
        # patch의 개수가 256개
        num_patch = patches.size(0)
        # 최종적으로 256 x 768 : linear projection에 들어가기 직전에 patch vector들의 모임
        return patches.reshape(num_patch,-1)


    

class Flattened2Dpatches:
    def __init__(self, patch_size=16, dataname='imagenet', img_size=256, batch_size=64): # 외부에서 정해줄 수 있는 변수들(하이퍼파라미터 ?!)
        self.patch_size = patch_size
        self.dataname = dataname
        self.img_size = img_size
        self.batch_size = batch_size

    def make_weights(self, labels, nclasses):
        labels = np.array(labels)
        weight_arr = np.zeros_like(labels)
        _, counts = np.unique(labels, return_counts=True)
        for cls in range(nclasses):
            weight_arr = np.where(labels == cls, 1/counts[cls], weight_arr) 
    
        return weight_arr 

    # main 함수
    # patchdata 함수는 Input 값이 없고, Dataname으로부터 데이터 결정됨.
    def patchdata(self):
        # CIFAR10에 대한 이미지 평균과 표준편차이다. (transform에 이용)
        # 다른 데이터도 훈련시키려면 mean, std, train_transform, test_transform 부분 각각의 데이터에다가 다르게 넣어주기 ! (일단 base로 CIFAR10만 훈련)
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        train_transform = transforms.Compose([transforms.Resize(self.img_size), transforms.RandomCrop(self.img_size, padding=2),
                                              transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std),
                                              PatchGenerator(self.patch_size)])
        test_transform = transforms.Compose([transforms.Resize(self.img_size), transforms.ToTensor(),
                                             transforms.Normalize(mean, std), PatchGenerator(self.patch_size)])

        if self.dataname == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
            # validation set을 구하기 위해서(간단하게 반반으로 구성)
            evens = list(range(0, len(testset), 2))
            odds = list(range(1, len(testset), 2))
            valset = torch.utils.data.Subset(testset, evens)
            testset = torch.utils.data.Subset(testset, odds)
          
        elif self.dataname == 'imagenet':
            pass

        # 배치를 만들 때 weightedRandomSampling 넣은 것
            # weightedRandomSampling : 각 클래스마다 전체 확률을 동일하게 하자는 개념
            # 예를 들어 4개의 클래스의 이미지 개수가 2, 2, 3, 3개가 있다고 하면은 원래는 2/10, 2/10, 3/10, 3/10의 확률로 뽑히지만
            # 확률을 조정해서(weighted) 2개가 있는 클래스는 각 1/8의 확률, 3개가 있는 클래스는 각 1/12의 확률로 만들어서 1/4의 확률로 같게 
            # 학습을 할 때는 한 배치당, 각 클래스가 동일한 개수가 들어올 수 있도록 weightedRandomSampling -> sampler 도입
        weights = self.make_weights(trainset.targets, len(trainset.classes))  # 가중치 계산
        weights = torch.DoubleTensor(weights) # 넘파이로 만들어 진 함수이니까 텐서로 바꿔준다.
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        trainloader = DataLoader(trainset, batch_size=self.batch_size, sampler=sampler)
        valloader = DataLoader(valset, batch_size=self.batch_size, shuffle=False)
        testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)

        return trainloader, valloader, testloader




# 얼마나 잘 나눠지는지 확인하기 위한 코드

def imshow(img):
    plt.figure(figsize=(100,100))
    # imshow 이용할 때 numpy로 바꿔서 계산
    plt.imshow(img.permute(1,2,0).numpy())
    plt.savefig('pacth_example.png')


if __name__ == "__main__":
    print("Testing Flattened2Dpatches..")
    batch_size = 64
    patch_size = 8
    img_size = 32
    num_patches = int((img_size*img_size)/(patch_size*patch_size))
    d = Flattened2Dpatches(dataname='cifar10', img_size=img_size, patch_size=patch_size, batch_size=batch_size)
    trainloader, _, _ = d.patchdata() # 나머지 _는 valloader, testloader, 이 데이터들은 flattened data이다.
    images, labels = iter(trainloader).next() # 샘플 확인할 때, iter-next
    print(images.size(), labels.size())

    # 예시로 첫번째 이미지만 확인
    # 아직까지 images가 flattened 데이터이므로 다시 사각형으로 만들기 위해서 reshape (사각형으로 만들어서 3채널 이미지를 만들기)
    sample = images.reshape(batch_size, num_patches, -1, patch_size, patch_size)[0] 
    print("Sample image size: ", sample.size())
    # make_grid : 나눴던 여러개의 이미지를 한번에 합쳐서 보여주는 역할
    imshow(torchvision.utils.make_grid(sample, nrow=int(img_size/patch_size)))
    
    # 주의할 점 : 위에서 train 이미지를 정규화했으므로, 다시 original 이미지를 출력하려면 
        # 처음에 정규화하는 과정을 주석처리하고 imshow 하던가
        # 마지막에 역으로 정규화하는 수식 넣던가 