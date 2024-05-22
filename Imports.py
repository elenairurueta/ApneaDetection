import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split, default_collate
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, recall_score, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import time
import seaborn as sns