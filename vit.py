"""
main.py와 같은 파일
"""
# 아래의 세개는 만든 모듈 
import patchdata
import model
import test
# 파이토치에서 제공되는 모듈
import torch
import torch.optim as optim
import torch.nn as nn
# parser 함수 이용하면 파이썬 파일을 직접 열지 않고 터미널에서 실행 가능하도록
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vision Transformer')
    # 하이퍼파라미터 나열
    parser.add_argument('--img_size', default=32, type=int, help='image size')
    parser.add_argument('--patch_size', default=4, type=int, help='patch size')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    # val acc를 가지고 early stopping을 할 건데 validation에 대해서 굳이 save할 필요는 없으므로 하한선 걸어줌 (50%가 안되면 save 안되게)
    parser.add_argument('--save_acc', default=50, type=int, help='val acc') 
    parser.add_argument('--epochs', default=501, type=int, help='training epoch')
    parser.add_argument('--lr', default=2e-3, type=float, help='learning rate')
    parser.add_argument('--drop_rate', default=.1, type=float, help='drop rate')
    # L2 regularization의 페널티
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    # 클래스의 개수에 따라서 MLP Head의 마지막 단에 node수가 정해짐
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
    # D (이미지 패치가 있고, 그것이 linear projection을 들어 갔을 때 나오는 vector dimension)
    parser.add_argument('--latent_vec_dim', default=128, type=int, help='latent dimension')
    # attention head 개수
    parser.add_argument('--num_heads', default=8, type=int, help='number of heads')
    parser.add_argument('--num_layers', default=12, type=int, help='number of layers in transformer')
    parser.add_argument('--dataname', default='cifar10', type=str, help='data name')
    # 대부분 파일이 크면, train/Inference하고 따로 파일 분리하는데 걍 같이 합침
    parser.add_argument('--mode', default='train', type=str, help='train or evaluation')
    parser.add_argument('--pretrained', default=False, type=bool, help='pretrained model')
    args = parser.parse_args()
    # parser에서 어떤 숫자들을 사용했는지 확인하기 위해 
    print(args)

    latent_vec_dim = args.latent_vec_dim
    # 인코더 내 MLP의 은닉층 노드 수(임의로 설정 가능)
    mlp_hidden_dim = int(latent_vec_dim/2) 
    num_patches = int((args.img_size * args.img_size) / (args.patch_size * args.patch_size))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Image Patches
    d = patchdata.Flattened2Dpatches(dataname=args.dataname, img_size=args.img_size, patch_size=args.patch_size,
                                     batch_size=args.batch_size)
    trainloader, valloader, testloader = d.patchdata()
    # 패치 사이즈(?x?x??..)를 알기 위해 iter-next 이용해서 patch 하나 불러옴
    image_patches, _ = iter(trainloader).next()

    # Model
    # 모델 불러오기
    vit = model.VisionTransformer(patch_vec_size=image_patches.size(2), num_patches=image_patches.size(1),
                                  latent_vec_dim=latent_vec_dim, num_heads=args.num_heads, mlp_hidden_dim=mlp_hidden_dim,
                                  drop_rate=args.drop_rate, num_layers=args.num_layers, num_classes=args.num_classes).to(device)

    if args.pretrained == 1 :
        # 경로도 argpaser에 넣어서 pretrained가 여러개 있으면 거기서 불러오고 싶은 모델명 선택 가능
        vit.load_state_dict(torch.load('./model.pth'))

    if args.mode == 'train':
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss() # Classification
        optimizer = optim.Adam(vit.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # optimizer = torch.optim.SGD(vit.parameters(), lr=args.lr, momentum=0.9) 논문에서는 모멘텀 사용
        # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(trainloader), epochs=args.epochs) 가능
            # 스케쥴링 종류 많음(코사인, 베이스, 스텝 등등)
        """
        if args.optim == "Adam" :
            optimizer = optim.Adam(vit.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optim = "Momentum" :
            optimizer = torch.optim.SGD(vit.parameters(), lr=args.lr, momentum=0.9)
        이런식으로 사용 가능
        """

        # Train
        # 배치에 대한 평균 loss를 계산해주기 위해 trainloader의 크기 = 배치의 개수(for문이 배치의 개수만큼 돈다.)
        n = len(trainloader)
        # save를 어느 accuracy 부터 할건지 ?
        best_acc = args.save_acc
        for epoch in range(args.epochs):
            # 초기값 0 설정
            running_loss = 0
            # flattened patch가 모델로 들어가면, linear projection을 거치고 positional embedding 해주고, transformer 인코더로 들어가서 12번 반복, MLP헤드 들어가서 output 출력 
            for img, labels in trainloader:   # flattened patch가 들어온다.
                optimizer.zero_grad()
                # flattened patch가 모델로 들어가면 
                # linear projection을 거치고, positional embedding 해주고, transformer encoder 반복, MLP Head 들어가서 ouput 출력
                outputs, _ = vit(img.to(device))  
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # scheduler.step() : 매 배치마다 반복
            # scheduler.step() : 매 epoch마다 반복

            train_loss = running_loss / n
            val_acc, val_loss = test.accuracy(valloader, vit)
            # if epoch % 5 == 0:
            print('[%d] train loss: %.3f, validation loss: %.3f, validation acc %.2f %%' % (epoch, train_loss, val_loss, val_acc))

            if val_acc > best_acc: 
                best_acc = val_acc
                print('[%d] train loss: %.3f, validation acc %.2f - Save the best model' % (epoch, train_loss, val_acc))
                torch.save(vit.state_dict(), './model.pth')

    else: 
        test_acc, test_loss = test.accuracy(testloader, vit)
        print('test loss: %.3f, test acc %.2f %%' % (test_loss, test_acc))
