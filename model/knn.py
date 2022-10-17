import math as mt

from utils.normalization import standardization


class TreeNode:
    """
    二叉树结点

    :attr
        left: 左子结点 \n
        right: 右子结点 \n
        is_accessed: 访问标记，用于记录是否被访问过 \n
        val: 结点值，存放训练实例 \n
    """

    left = None
    right = None
    is_accessed = False
    val = 0

    def __init__(self, val):
        self.val = val


def get_balanced_kdtree(train_datas, depth=0):
    """
    构造平衡 kd 树

    :param train_datas: 训练实例集，带有标记，二维数组，形如 [[1, 2, 3, ..., "A"], [2, 3, 4, ..., "B"], ...]
    :param depth: 相对于初始根结点，此时的深度，整数

    :return 平衡 kd 树的根结点，TreeNode
    """

    if len(train_datas) == 0:
        return None

    # 确定当前维度以及当前维度序列的中位数索引
    # 最后一维为标记，不参与构造时的比较
    cur_dim = depth % (len(train_datas[0]) - 1)
    mid_index = mt.floor(len(train_datas) / 2)

    # 对当前维度的序列进行排序
    sorted_list = train_datas.copy()
    sorted_list.sort(key=lambda item: item[cur_dim])

    # 递归构造二叉树
    root = TreeNode(sorted_list[mid_index])
    root.left = get_balanced_kdtree(sorted_list[:mid_index], depth + 1)
    root.right = get_balanced_kdtree(sorted_list[mid_index + 1:], depth + 1)

    return root


def get_knn(sample, classifier_info, parent=None, distances=[], depth=0):
    """
    计算 knn，k 个最近邻点

    :param sample: 实例，一维数组, 形如 [1, 2, 3, ...]，存有特征
    :param classifier_info: 分类器信息，字典，形如 { "k": 1, "balanced_kdtree": TreeNode(), },
     k: 取最近邻点的数量，整数, balanced_kdtree: 平衡 kd 树，TreeNode
    :param parent: 此层节点的父结点，TreeNode
    :param distances: 距离集，二维数组，形如 [[1, train_data]，[2, train_data], ...]
    :param depth: 相对于初始根结点，此时的深度，整数

    :return 已排序的 k 个最近邻点集，二维数组, 形如 [[123.34, TreeNode().val], ...], knn[0][0] 为距离，knn[0][1] 为实例
    """

    k, balanced_kdtree = classifier_info["k"], classifier_info["balanced_kdtree"]

    if balanced_kdtree is None:
        return

    # 设置访问标记
    balanced_kdtree.is_accessed = True

    # 计算当前用于比较的维度
    cur_dim = depth % (len(sample))

    if sample[cur_dim] <= balanced_kdtree.val[cur_dim]:
        get_knn(sample, {"k": k, "balanced_kdtree": balanced_kdtree.left}, balanced_kdtree, distances, depth + 1)
    else:
        get_knn(sample, {"k": k, "balanced_kdtree": balanced_kdtree.right}, balanced_kdtree, distances, depth + 1)

    # 目标实例与当前结点的距离
    cur_distance = 0

    for i in range(len(sample)):
        cur_distance += (sample[i] - balanced_kdtree.val[i]) ** 2

    distances.append([mt.sqrt(cur_distance), balanced_kdtree.val])

    # 判断兄弟结点超矩形是否和当前结点与目标结点构成的超球形有交点
    if parent is not None and parent.left is not None and parent.right is not None:
        # 目标实例与当前结点的兄弟结点所在超矩形的距离，
        another_distance = (sample[cur_dim - 1] - parent.val[cur_dim - 1]) ** 2

        if another_distance < cur_distance:
            if parent.left is balanced_kdtree and parent.right.is_accessed is True:
                get_knn(sample, {"k": k, "balanced_kdtree": parent.left}, parent, distances, depth)
            elif parent.right is balanced_kdtree and parent.left.is_accessed is True:
                get_knn(sample, {"k": k, "balanced_kdtree": parent.right}, parent, distances, depth)

    knn = sorted(distances, key=lambda distance: distance[0])[:k]

    return knn


class Classifier:
    """
    分类器

    :attr
        balanced_kdtree: 构造好的平衡 kd 树 \n
        k: 取最近邻点的数量 \n
        classify(sample): 对 sample 进行分类，得到分类结果
    """

    balanced_kdtree = None
    k = 0
    avgs = []
    std_devs = []

    def __init__(self, balanced_kdtree, k, avgs_and_std_devs):
        self.balanced_kdtree = balanced_kdtree
        self.k = k
        self.avgs, self.std_devs = avgs_and_std_devs["avgs"], avgs_and_std_devs["std_devs"]

    def classify(self, sample):
        """
        使用 knn 方法分类

        :param sample: 实例，一维数组, 形如 [1, 2, 3, ...]，存有特征

        :return 字典，存有‘type’、‘knn’、’prob‘三个键，它们的值分别代表预测的类别，k 个最近邻点集，预测得到的类别的可能性
        """

        normed_data = standardization.get_normed_data(sample, {"avgs": self.avgs, "std_devs": self.std_devs})

        # knn 形如 [[distance, TreeNode().val], ...]
        knn = get_knn(normed_data, {"k": self.k, "balanced_kdtree": self.balanced_kdtree})
        record = dict()

        # 记录次数，多数表决
        # record 形如 {"A": 0.3435, B": 0.6544, ...}
        # e 形如 [123.23, TreeNode().val]
        # 1 / e[0] 为以距离倒数为权重，越近的点对结果影响越大
        for e in knn:
            if e[1][-1] in record:
                record[e[1][-1]] = record[e[1][-1]] + 1 / e[0]
            else:
                record[e[1][-1]] = 1 / e[0]

        # items 形如 [["A", 0.3435], ["A", 0.6544], ...]
        items = list(record.items())
        items.sort(key=lambda elem: elem[1])

        # 概率
        prob = 0

        for v in record.values():
            prob += v

        prob = items[-1][1] / prob

        for i in range(len(knn)):
            knn[i] = knn[i][1]

        res = {
            'type': items[-1][0],
            'knn': knn,
            'prob': prob,
        }

        return res


def get_signs_appended_dataset(train_datas_x, train_datas_y):
    """
    获得添加标记后的数据集

    :param train_datas_x: 训练实例集
    :param train_datas_y: 训练实例标记集

    :return: 添加标记后的训练实例集
    """

    train_datas = train_datas_x.copy()

    for i in range(len(train_datas)):
        train_datas[i].append(train_datas_y[i])

    return train_datas


def get_handled_dataset(dataset_info):
    """
    获得对原始数据集处理后的数据集，包括添加标记和归一化操作

    :param dataset_info: 数据集信息，字典，形如 { "train_datas": [[1, 2, 3, ...], [1, 2, 3, ...], ...],
    "signs": [1, 2, 3, ...], "avgs_and_std_devs": {"avgs":[1, 2, 3, ...], "std_devs": [1, 2, 3, ...]} }

    :return:
    """

    handled_dataset = []

    train_dataset = dataset_info['train_dataset']
    signs = dataset_info['signs']
    avgs_and_std_devs = dataset_info['avgs_and_std_devs']

    for ele in train_dataset:
        normed_data = standardization.get_normed_data(ele, avgs_and_std_devs)
        handled_dataset.append(normed_data)

    handled_dataset = get_signs_appended_dataset(handled_dataset, signs)

    return handled_dataset


def test():
    k = 2
    train_dataset = [[1, 1, 1, 1],
                     [0.5, 1, 1, 1],
                     [0.1, 0.1, 0.1, 0.1],
                     [0.5, 0.5, 0.5, 0.5],
                     [1, 0.8, 0.3, 1],
                     [0.6, 0.5, 0.7, 0.5],
                     [1, 1, 0.9, 0.5],
                     [1, 0.6, 0.5, 0.8],
                     [0.5, 0.5, 1, 1],
                     [0.9, 1, 1, 1],
                     [0.6, 0.6, 1, 0.1],
                     [1, 0.8, 0.5, 0.5],
                     [1, 0.1, 0.1, 1],
                     [1, 1, 0.7, 0.3],
                     [0.2, 0.3, 0.4, 0.5],
                     [0.5, 1, 0.6, 0.6]
                     ]

    signs = ['女神',
             '淑女',
             '丑女',
             '一般型',
             '淑女',
             '一般型',
             '女神',
             '一般型',
             '淑女',
             '女神',
             '丑女',
             '可爱型',
             '可爱型',
             '淑女',
             '丑女',
             '可爱型'
             ]

    sample = [1, 1, 0.9, 1]

    avgs_and_std_devs = standardization.get_avgs_and_std_devs(train_dataset)
    handled_dataset = get_handled_dataset({
        "train_dataset": train_dataset,
        "signs": signs,
        "avgs_and_std_devs": avgs_and_std_devs
    })

    cls_1 = Classifier(get_balanced_kdtree(handled_dataset), k, avgs_and_std_devs)
    res = cls_1.classify(sample)

    print("The k nearest neighbors of the sample are:\n", res['knn'])
    print()
    print("The sample is forecasted as type ", res['type'])
    print("The probability of the forecast type is ", res['prob'])


if __name__ == '__main__':
    test()
