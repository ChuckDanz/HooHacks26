import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '../CatVTON')
print('1 importing torch...')
import torch
print('2 torch ok, cuda:', torch.cuda.is_available())
print('3 importing CatVTON pipeline...')
from model.pipeline import CatVTONPipeline
print('4 pipeline ok')
from model.cloth_masker import AutoMasker
print('5 masker ok')
from size_pipeline import SizeVariablePipeline
print('6 size_pipeline ok')
print('ALL IMPORTS OK')
