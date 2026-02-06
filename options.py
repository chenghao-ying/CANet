import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--trainsize', type=int, default=512, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')  # 100
parser.add_argument('--load_pre', type=str, default='./smt_base.pth', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--rgb_root', type=str, default='./COD-TrainDataset/Imgs/', help='the training rgb images root')  # train_dut
parser.add_argument('--depth_root', type=str, default='./COD-TrainDataset/depth/', help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='./COD-TrainDataset/GT/', help='the training gt images root')
parser.add_argument('--test_rgb_root', type=str, default='./COD-TestDataset/COD10K/Imgs/', help='the test gt images root')
parser.add_argument('--test_depth_root', type=str, default='./COD-TestDataset/COD10K/depth/', help='the test gt images root')
parser.add_argument('--test_gt_root', type=str, default='./COD-TestDataset/COD10K/GT/', help='the test gt images root')

# parser.add_argument('--rgb_root', type=str, default='./RGBT_dataset/train/RGB/', help='the training rgb images root')  # train_dut
# parser.add_argument('--depth_root', type=str, default='./RGBT_dataset/train/T/', help='the training depth images root')
# parser.add_argument('--gt_root', type=str, default='./RGBT_dataset/train/GT/', help='the training gt images root')
# parser.add_argument('--test_rgb_root', type=str, default='./RGBT_dataset/val/RGB/', help='the test gt images root')
# parser.add_argument('--test_depth_root', type=str, default='./RGBT_dataset/val/T/', help='the test gt images root')
# parser.add_argument('--test_gt_root', type=str, default='./RGBT_dataset/val/GT/', help='the test gt images root')


# parser.add_argument('--rgb_root', type=str, default='./SOD_DATA/train_dut/train_images/', help='the training rgb images root')  # train_dut
# parser.add_argument('--depth_root', type=str, default='./SOD_DATA/train_dut/train_depth/', help='the training depth images root')
# parser.add_argument('--gt_root', type=str, default='./SOD_DATA/train_dut/train_masks/', help='the training gt images root')
# parser.add_argument('--test_rgb_root', type=str, default='./SOD_DATA/test_datasets/NJU2K/RGB/', help='the test gt images root')
# parser.add_argument('--test_depth_root', type=str, default='./SOD_DATA/test_datasets/NJU2K/depth/', help='the test gt images root')
# parser.add_argument('--test_gt_root', type=str, default='./SOD_DATA/test_datasets/NJU2K/GT/', help='the test gt images root')

parser.add_argument('--save_path', type=str, default='./cpts/', help='the path to save models and logs')
opt = parser.parse_args()