"""
추가로 test 결과에 대한 visualization 할 수 있는 함수도 추가하면 좋을 듯 !
또다른 test에 필요한 함수 추가
"""

import torch
import torch.nn as nn

def accuracy(dataloader, model):
    correct = 0
    total = 0
    running_loss = 0
    n = len(dataloader)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        # ViT는 drop out을 사용하므로 test 시에는 비활성화 해주어야 하기 때문에 model.eval() 실행
        model.eval()
        # val or test set 들어옴
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device) # GPU 연산(반드시 붙여주기)
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item() 
            running_loss += loss.item()

        # test 시에는 업데이트를 안하기 때문에 loss function 필요 없지만, loss 값 알아보기 위해 넣었음. (계산만 하고 업데이트 X)
        loss_result = running_loss / n

    acc = 100 * correct / total
    # 나갈때는 훈련용으로 다시 바꾸기 ! 
    model.train()
    return acc, loss_result


