# Custom loss function
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
    
    def forward(self, predictions, targets, mask):
        squared_errors = (predictions - targets) ** 2
        masked_errors = squared_errors * mask
        sum_errors = torch.sum(masked_errors)
        sum_mask = torch.sum(mask)
        
        return sum_errors / (sum_mask + 1e-8)
