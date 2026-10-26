import argparse
import time
import warnings
import pandas as pd
from torch.utils.data import DataLoader
from data_loader import TrainLoader, load_feature, DevelLoader, Loader
from model_trainer import ModelTrainer
from utils import init_args, cm_analysis

warnings.simplefilter("ignore")


def main(args):
    """
    模型训练主函数
    Parameter：
        args:参数配置项
    """

    # 加载数据
    if args.feature_type:
        feat_df = load_feature(args.feature_type, args.feature_path)
    else:
        feat_df = pd.DataFrame()

    src_dataset = TrainLoader(feat_df, **vars(args))
    target_dataset = Loader(feat_df, **vars(args))
    devel_dataset = DevelLoader(feat_df, **vars(args))
    train_loader = DataLoader(src_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.n_cpu,
                              drop_last=False,
                              pin_memory=True
                              )
    loader = DataLoader(target_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.n_cpu,
                        drop_last=False,
                        pin_memory=True
                        )
    devel_loader = DataLoader(devel_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.n_cpu,
                              drop_last=False,
                              pin_memory=True
                              )
    # 初始化trainer
    trainer = ModelTrainer(**vars(args))

    # 记录结果
    uars = [0]
    losses = [0]
    score_file = open(args.score_save_path, "a+")
    score_file.write(f'Model: {args.model} \n'
                     f'Feature Type: {args.feature_type} \n'
                     + '-' * 20 + '\n')

    # 模型训练
    for epoch in range(1, args.epoch + 1):
        print("-" * 10, f'第{epoch}轮训练开始', "-" * 10)

        # 训练
        lr, loss = trainer.train_network(epoch, train_loader, loader)
        losses.append(loss)
        # 测试
        if epoch % args.test_step == 0:
            uar, true_list, pred_list, pred_num = trainer.eval_network(devel_loader)
            if uar >= max(uars):
                trainer.save_parameters(args.model_save_path + "/model_best_%6f_0.pkl" % lr)
                cm_analysis(lr, true_list, pred_list, pred_num, args.save_path + '/cm',
                            sorted(set(true_list)))
            uars.append(uar)

            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                  "%d epoch, LOSS %f, UAR %3.3f, bestUAR %3.3f" % (epoch, loss, uars[-1], max(uars)))

            score_file.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                             "—— %d epoch, LR %f, LOSS %f, UAR %3.3f,  bestUAR %3.3f\n" % (
                                 epoch, lr, loss, uars[-1], max(uars)))
            score_file.flush()

    score_file.write('\n')
    score_file.close()


if __name__ == '__main__':
    epoch = 30
    lr = 0.0005

    parser = argparse.ArgumentParser(description='Recognition')

    # 训练参数设置
    parser.add_argument('--epoch', type=int, default=epoch, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--n_cpu', type=int, default=2, help='Number of loader threads')
    parser.add_argument('--test_step', type=int, default=1, help='Test and save every [test_step] epochs')

    # 训练、开发、测试集路径, save_path
    parser.add_argument('--wav_path', type=str, default='E:/work/USERNAME/DEMOS/wav_all/diff/wav',
                        help='Path of the WAV file')
    parser.add_argument('--train_path', type=str, default='E:/work/USERNAME/DEMOS/lab/diff/train.csv',
                        help='Path of the train label file')
    parser.add_argument('--target_path', type=str, default='E:/work/USERNAME/DEMOS/lab/same/train.csv',
                        help='Path of the train label file')
    parser.add_argument('--devel_path', type=str, default='E:/work/USERNAME/DEMOS/lab/diff/devel.csv',
                        help='Path of the devel label file')
    parser.add_argument('--test_path', type=str, default='E:/work/USERNAME/DEMOS/lab/diff/test.csv',
                        help='Path of the test label file')
    parser.add_argument('--save_path', type=str, default="exps/DEMOS/diff",
                        help='Path to save the score.txt and models')

    # 选用已有特征集

    parser.add_argument('--feature_type', type=str, default='', help='选择已有的特征集, 默认None表示不用')
    parser.add_argument('--feature_path', type=str, default='E:/work/USERNAME/DEMOS/features/diff/it/',
                        help='特征集位置')
    # 模型参数

    parser.add_argument('--model', type=str, default='BSP', help='model')
    parser.add_argument('--lr', type=float, default=lr, help='Learning rate')
    parser.add_argument("--lr_decay", type=float, default=1.0, help='Learning rate decay every [test_step] epochs')

    args = parser.parse_args()
    args = init_args(args)

    st = time.time()
    main(args)
    print(f'模型训练结束，共耗时：{round(time.time() - st, 2)}s')
