import os
import cv2
from tqdm import tqdm
from py_sod_metrics import Smeasure, Emeasure, WeightedFmeasure, MAE

# gt_path = 'G:/DataSet/CAMO/GT/'
# gt_path = 'G:/DataSet/CHAMELEON_TestingDataset/GT/'
gt_path = 'G:/DataSet/COD10K-v3/Test/GT_Object/'
predict_path = './results/COD10K/'

mae = MAE()
wfm = WeightedFmeasure()
sm = Smeasure()
em = Emeasure()

images = os.listdir(predict_path)
for image in tqdm(images):
    gt = cv2.imread(os.path.join(gt_path, image), 0)
    predict = cv2.imread(os.path.join(predict_path, image), 0)

    h, w = gt.shape
    predict = cv2.resize(predict, (w, h))

    mae.step(predict, gt)
    wfm.step(predict, gt)
    sm.step(predict, gt)
    em.step(predict, gt)

print('mae: %.4f' % mae.get_results()['mae'])
print('wfm: %.4f' % wfm.get_results()['wfm'])
print('em: %.4f' % em.get_results()['em']['curve'].mean())
print('sm: %.4f' % sm.get_results()['sm'])

# camo          0.8435 0.8949 0.7746 0.0629
# chameleon     0.8974 0.9497 0.8350 0.0270
# cod10k        0.8404 0.9187 0.7288 0.0297
