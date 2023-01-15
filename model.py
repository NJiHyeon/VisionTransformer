import torch
import torch.nn as nn


class LinearProjection(nn.Module):
    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, drop_rate):
        super().__init__()
        self.linear_proj = nn.Linear(patch_vec_size, latent_vec_dim)
        # 학습 가능해야 하므로 파라미터로 정의
        # nn.Parameter : 모델 업데이트를 할 때, 같이 업데이트 되는 변수 
        self.cls_token = nn.Parameter(torch.randn(1, latent_vec_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, latent_vec_dim))
        self.dropout = nn.Dropout(drop_rate)

    """
    <def forward( )>
    B x N x P^2C 
    -> linear projection
    B x N x D 
    -> class token(1xD) 추가해야 하는데 크기를 맞춰주어야 하므로
    -> repeat 함수를 써서 class token 차원 맞춰주기 (B x 1 x D)
    -> concatenate x랑 cls_token이랑 dim=1 기준이므로 N+1 
    B x (N+1) x D
    """
    def forward(self, x):
        batch_size = x.size(0)
        # 각각의 배치에 동일한 클래스 토큰 -> 반복해서 차원 맞춰주기
        x = torch.cat([self.cls_token.repeat(batch_size, 1, 1), self.linear_proj(x)], dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        # transformer의 input 완성
        return x  



class MultiheadedSelfAttention(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, drop_rate):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.latent_vec_dim = latent_vec_dim
        self.head_dim = int(latent_vec_dim / num_heads)
        
        # 또 다른 방법 가능 
            # self.all = nn.Linear(latent_vec_dim, 3*latent_vec_dim)
            # forward에서 q, k, v 다시 떼주기
        # 사실 원래 query dimension : D_h
            # D_h = D/k(이렇게 하고 뒤에서 multi-head 해줘야함) 
            # D = D_h * k (즉, D_h를 head 수만큼 계산했다는 얘기)
            # 따라서 한번에 모든 Head의 query, key, value를 계산한 셈
        # latent_vec_dim = D = k * D_h니까 k, D_h를 나눠주어야 Head 마다 각 q, k, v가 나오므로
            # forward에서 .view를 통해 재정비 
            
        self.query = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.key = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.value = nn.Linear(latent_vec_dim, latent_vec_dim)
        # 학습이 되면 안되니까 그냥 텐서로만 정의
        # (단, 텐서 하나를 만들면 cpu 연산만 되는 텐서이므로 gpu 연산이 될 수 있도록 to(device) 붙여주기
        self.scale = torch.sqrt(self.head_dim*torch.ones(1)).to(device)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # (batch_size, n(벡터의 개수), head의 개수, head 차원)
            # head 차원 * head의 개수 = latent_vec_dim(D)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,3,1) # k.T
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        # 메트릭스 곱은 @로 처리 가능(=torch.matmul)
        attention = torch.softmax(q @ k / self.scale, dim=-1)
        # 정규화
        x = self.dropout(attention) @ v 
        # concatenate 해주어야 하기 때문에 reshape
        x = x.permute(0,2,1,3).reshape(batch_size, -1, self.latent_vec_dim)

        return x, attention

# 레이어 하나에 해당하는 기능
class TFencoderLayer(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate):
        super().__init__()
        self.ln1 = nn.LayerNorm(latent_vec_dim)
        self.ln2 = nn.LayerNorm(latent_vec_dim)
        self.msa = MultiheadedSelfAttention(latent_vec_dim=latent_vec_dim, num_heads=num_heads, drop_rate=drop_rate)
        self.dropout = nn.Dropout(drop_rate)
        # mlp : 층이 2개, activation function(GeLU)
        self.mlp = nn.Sequential(nn.Linear(latent_vec_dim, mlp_hidden_dim),
                                 nn.GELU(), nn.Dropout(drop_rate),
                                 nn.Linear(mlp_hidden_dim, latent_vec_dim),
                                 nn.Dropout(drop_rate))

    def forward(self, x):
        z = self.ln1(x)
        z, att = self.msa(z)
        z = self.dropout(z)
        x = x + z
        z = self.ln2(x)
        z = self.mlp(z)
        x = x + z

        return x, att



class VisionTransformer(nn.Module):
    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate, num_layers, num_classes):
        super().__init__()
        """
        1) self.patchembedding
            - 들어온 데이터에 대해서 linear projection을 해줘야 하니까 patch_embedding이라는 이름으로 정의
            - linear projection 후, 클래스 토큰 넣고, patchembedding까지 해주는 작업
            - patch_vec_size = P^2 * C
            - num_patches = HW / p^2
            - latent_vec_dim = D
            - drop_rate 추가 
            
        2) self.transformer
            - 트랜스포머가 레이어를 반복하니까, 그 반복을 리스트 안에다 넣을 것 (for문을 사용해서 layer 수만큼 반복해서 append)
            - 레이어 수가 12면, 각각의 클래스를 12번 선언한 것이기 때문에 각 레이어는 파라미터 공유하지 않고 독립적
            - ModuleList라는 곳에 넣어서 학습을 할 수 있도록 한다.  
        
        3) mlp_head
        - 트랜스포머 인코더로부터 나온 결괏값에, 첫번째 클래스 토큰 부분에 해당되는 벡터만 뽑아서 분류 실시
        - latent_vec_dim(D) 하나짜리 들어가서 num_classes 수만큼 출력
        """
        
        # 들어온 데이터에 대해서 linear projection을 해줘야 하니까 patch_embedding이라는 이름으로 정의
        self.patchembedding = LinearProjection(patch_vec_size=patch_vec_size, num_patches=num_patches,
                                               latent_vec_dim=latent_vec_dim, drop_rate=drop_rate)
        self.transformer = nn.ModuleList([TFencoderLayer(latent_vec_dim=latent_vec_dim, num_heads=num_heads,
                                                         mlp_hidden_dim=mlp_hidden_dim, drop_rate=drop_rate)
                                          for _ in range(num_layers)])

        self.mlp_head = nn.Sequential(nn.LayerNorm(latent_vec_dim), nn.Linear(latent_vec_dim, num_classes))


    def forward(self, x):
        # 학습, 평가에는 필요없긴 하지만 attention 값을 저장하고 싶을 때 
        att_list = []
        # patchembedding & positionalembedding까지 완료한 것 (실질적인 Input)
        x = self.patchembedding(x)
        # 리스트로 쌓여있는 것에서 하나씩 불러와서 계산(레이어 수만큼)
        for layer in self.transformer:
            x, att = layer(x)
            # 각 층마다 나오는 attention 저장 -> visualization에서 attention 그림 확인 가능
            att_list.append(att)
        # layer를 다 거치고 난 다음에 가장 앞부분 (class token)에 해당되는 부분의 벡터만 떼서 mlp_head 넣기
        x = self.mlp_head(x[:,0])

        # 최종 결과값 x, attention list 산출
        return x, att_list
