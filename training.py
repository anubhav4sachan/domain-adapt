import torch
import torch.optim

from net import mainNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(dataloader_source, dataloader_target):
    model = mainNet().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    criteria1 = torch.nn.CrossEntropyLoss()
    criteria2 = torch.nn.MSELoss()
    
    for epoch in range (30):
        model.train()
        running_loss = 0
        
        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)
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
            
            dom_label = torch.zeros(lbl_size).long().to(device)
            
            fout = model(inputs)[0]
            
            dout = model(inputs)[1]
            
            data_target = data_target_iter.next()
            inputs_t, lbl_t = data_target   
            inputs_t = inputs_t.cuda()
            lbl_t = lbl_t.cuda()
            
            lbl__tf = torch.Tensor.float(lbl_t)            
            lbl_size_t = len(lbl_t)
            
            dom_label_t = torch.ones(lbl_size_t).long().to(device)
                       
            dout_t = model(inputs_t)[1]
            
            loss = criteria1(dout, dom_label) + criteria2(fout, li) + criteria1(dout_t, dom_label_t)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i%200 == 0:
                print(epoch, i, loss.item())
                
            i += 1
                
    torch.save(model, 'mod.pth')

    return model