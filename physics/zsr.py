import torch
import logging

class ZAxisSuperResolution():
    def __init__(self, factor: int):
        self.factor = factor
    
    def A(self, x: torch.Tensor) -> torch.Tensor:
        N, C, Y, Z = x.shape

        assert C == 1
        if Z % self.factor != 0:
            # logging.warning(f"Z({Z}) % factor({self.factor}) != 0")
            x = x[..., 0:Z // self.factor * self.factor]

        Z_new = Z // self.factor

        result = torch.zeros((N, C, Y, Z_new), device=x.device)
        for i in range(self.factor):
            result += x[..., i::self.factor]
        result /= self.factor

        return result
    
    def A_T(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.factor
    
    def A_dagger(self, x:torch.Tensor):
        N, C, Y, Z = x.shape
        assert C == 1

        x = x.clone().detach()
        result = x.repeat_interleave(self.factor, dim=3)
        
        return result 