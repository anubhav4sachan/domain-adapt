import torch
import torch.optim

from net import mainNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_deep(dataloader_source):
    model = mainNet().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    criteria1 = torch.nn.CrossEntropyLoss()
    criteria2 = torch.nn.MSELoss()
    
    num_ep = 1
    
    for epoch in range (num_ep):
        model.train()
        running_loss = 0
        
#        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        len_dataloader = len(dataloader_source)
        data_source_iter = iter(dataloader_source)
#        data_target_iter = iter(dataloader_target)
        i = 0
        while i < len_dataloader:
            
            data_source = data_source_iter.next()
            inputs, lbl, li = data_source
            inputs = inputs.cuda()
            lbl = lbl.cuda()
            li = li.cuda()
            
            lbl_f = torch.Tensor.float(lbl)            
            lbl_size = len(lbl)
            
            optimizer.zero_grad()
            fout = model(inputs)[0]
            
            loss = criteria2(fout, li)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i%50 == 0:
                print(epoch, i, loss.item())
                
            i += 1
                
        torch.save(model, 'deep-model-'+str(epoch)+'.pth')

    return model